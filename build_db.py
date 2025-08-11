import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import chromadb
from pathlib import Path
from collections import Counter
import warnings
from sklearn.metrics.pairwise import cosine_distances
from collections import defaultdict

warnings.filterwarnings("ignore")

def load_data(json_path="train.json", npz_dir="../../reg2025/gigapath_vectors"):
    """JSON과 NPZ 파일 로딩"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    npz_dir = Path(npz_dir)
    valid_pairs = []
    
    for item in data:
        if not all(k in item for k in ['id', 'report', 'label']):
            continue
            
        item_id = item['id'].replace('.tiff', '').replace('.png', '').replace('.jpg', '')
        npz_path = npz_dir / f"{item_id}.npz"
        
        if npz_path.exists():
            valid_pairs.append((item, str(npz_path)))
    
    print(f"Found {len(valid_pairs)} valid pairs")
    return valid_pairs

def extract_embeddings(valid_pairs, seed=1417):
    """임베딩 추출 - 각 NPZ에서 5개 variation"""
    np.random.seed(seed)
    
    embedding_keys = [
        'embedding_macenko_original',
        'embedding_macenko_hsv', 
        'embedding_macenko_rgb_shift',
        'embedding_macenko_brightness_contrast',
        'embedding_macenko_he_aug'
    ]
    
    indices = np.random.permutation(len(valid_pairs))
    
    embeddings, reports, labels, item_ids = [], [], [], []
    
    for i in indices:
        item, npz_path = valid_pairs[i]
        
        try:
            data = np.load(npz_path, allow_pickle=True)
            
            for j, key in enumerate(embedding_keys):
                if key in data:
                    emb = data[key]
                    if len(emb.shape) == 1:
                        embeddings.append(emb)
                    elif len(emb.shape) == 2:
                        embeddings.append(emb[0])
                    else:
                        embeddings.append(emb.flatten())
                    
                    reports.append(item['report'])
                    labels.append(item['label'])
                    item_ids.append(f"{item['id']}_v{j}")
            
            data.close()
            
        except Exception as e:
            print(f"Error: {npz_path} - {e}")
    
    if len(embeddings) == 0:
        print("❌ No embeddings extracted")
        return None, None, None, None
    
    embeddings = torch.tensor(embeddings, dtype=torch.float32)
    print(f"✅ Extracted embeddings: {embeddings.shape}")
    print(f"   - {len(indices)} files × {len(embedding_keys)} variations = {len(embeddings)} total")
    return embeddings, reports, labels, item_ids

@staticmethod
def get_bleu4(ref_text, hyp_text):
    ref_words = ref_text.split()
    hyp_words = hyp_text.split()

    ref_fourgrams = [' '.join(ref_words[i:i+4]) for i in range(len(ref_words)-3)]
    hyp_fourgrams = [' '.join(hyp_words[i:i+4]) for i in range(len(hyp_words)-3)]

    count = 0
    total = 0

    for fourgram in hyp_fourgrams:
        count += min(hyp_fourgrams.count(fourgram), ref_fourgrams.count(fourgram))
        total += 1

    if total == 0:
        return 0.0

    precision = count / total
    return precision

@staticmethod
def get_rouge(ref_text, hyp_text):
    def lcs(X, Y):
        m = len(X)
        n = len(Y)
        L = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if X[i-1] == Y[j-1]:
                    L[i][j] = L[i-1][j-1] + 1
                else:
                    L[i][j] = max(L[i-1][j], L[i][j-1])
        return L[m][n]
    
    ref_tokens = ref_text.lower().split()
    hyp_tokens = hyp_text.lower().split()

    lcs_length = lcs(ref_tokens, hyp_tokens)

    ref_length = len(ref_tokens)
    hyp_length = len(hyp_tokens)

    precision = lcs_length / hyp_length if hyp_length > 0 else 0
    recall = lcs_length / ref_length if ref_length > 0 else 0

    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return f1_score

def compute_label_similarity(label1, label2):
    """라벨 유사도 계산"""
    tokens1 = label1.split('_')
    tokens2 = label2.split('_')
    
    if tokens1[0] != tokens2[0]:
        return 0.0
    
    hist_tokens1 = tokens1[2:]
    hist_tokens2 = tokens2[2:]
    
    if hist_tokens1 == hist_tokens2:
        bleu_score = 1.0
    elif len(hist_tokens1) < 4 or len(hist_tokens2) < 4:
        bleu_score = 0.0
    else:
        hist_type1 = ' '.join(hist_tokens1)
        hist_type2 = ' '.join(hist_tokens2)
        bleu_score = get_bleu4(hist_type1, hist_type2)
    
    hist_type1 = ' '.join(hist_tokens1)
    hist_type2 = ' '.join(hist_tokens2)
    rouge_score = get_rouge(hist_type1, hist_type2)
    
    return ((bleu_score + rouge_score) / 2) ** 2

def create_similarity_matrix(labels):
    """유사도 매트릭스 생성"""
    unique_labels = list(set(labels))
    sim_dict = {}
    
    for i, l1 in enumerate(unique_labels):
        for j, l2 in enumerate(unique_labels):
            if i <= j:
                sim = compute_label_similarity(l1, l2)
                sim_dict[(l1, l2)] = sim_dict[(l2, l1)] = sim
    
    return sim_dict

class SemanticContrastiveLoss(nn.Module):
    def __init__(self, temp=0.07, organ_weight=2.0, fine_weight=4.0):
        super().__init__()
        self.temp = max(temp, 1e-8)
        self.organ_w = organ_weight
        self.fine_w = fine_weight
    
    def forward(self, features, batch_labels, sim_dict):
        bs = features.shape[0]
        device = features.device
        
        if torch.isnan(features).any() or torch.isinf(features).any():
            print("❌ NaN/Inf in features")
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        sim_matrix = torch.zeros(bs, bs, device=device)
        for i in range(bs):
            for j in range(bs):
                sim_matrix[i, j] = sim_dict.get((batch_labels[i], batch_labels[j]), 0.0)
        
        cosine_sim = torch.matmul(features, features.T) / self.temp
        
        if torch.isnan(cosine_sim).any() or torch.isinf(cosine_sim).any():
            print("❌ NaN/Inf in cosine_sim")
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        mask = torch.eye(bs, dtype=torch.bool, device=device)
        cosine_sim = cosine_sim.masked_fill(mask, -1e9)
        
        organ_pos = (sim_matrix > 0.0) & (sim_matrix < 1.0)
        exact_pos = (sim_matrix == 1.0)
        
        organ_pos = organ_pos.float().masked_fill(mask, 0)
        exact_pos = exact_pos.float().masked_fill(mask, 0)
        
        has_pos = ((organ_pos + exact_pos).sum(1) > 0)
        if has_pos.sum() == 0:
            print("⚠️ No positive pairs found")
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        logits_max = torch.max(cosine_sim, dim=1, keepdim=True)[0]
        logits = cosine_sim - logits_max.detach()
        logits = torch.clamp(logits, min=-50, max=50)
        
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        
        if torch.isnan(log_prob).any():
            print("❌ NaN in log_prob")
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        weighted_pos = (organ_pos * sim_matrix * self.fine_w + exact_pos * self.organ_w)
        pos_sum = weighted_pos.sum(1)
        pos_sum = torch.clamp(pos_sum, min=1e-8)
        
        mean_log_prob = (weighted_pos * log_prob).sum(1) / pos_sum
        loss = -mean_log_prob[has_pos].mean()
        
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"❌ Final loss is NaN/Inf")
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        return loss

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim)
        )
    
    def forward(self, x):
        return F.normalize(self.net(x), p=2, dim=1)

def train_model(embeddings, labels, epochs=10, batch_size=64):
    """모델 학습"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print(f"📊 Input embeddings shape: {embeddings.shape}")
    print(f"   - Min/Max/Mean: {embeddings.min():.4f}/{embeddings.max():.4f}/{embeddings.mean():.4f}")
    
    if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
        print("❌ NaN/Inf in input embeddings")
        return None, None
    
    sim_dict = create_similarity_matrix(labels)
    print(f"Created similarity matrix with {len(sim_dict)} pairs")
    
    model = ProjectionHead(embeddings.shape[1]).to(device)
    print(f"📊 Model input dim: {embeddings.shape[1]} -> output dim: 128")
    
    criterion = SemanticContrastiveLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    unique_labels = list(set(labels))
    label_idx = {l: [i for i, x in enumerate(labels) if x == l] for l in unique_labels}
    
    print(f"Unique labels: {len(unique_labels)}")
    
    model.train()
    best_loss = float('inf')
    
    for epoch in range(epochs):
        total_loss, n_batches = 0, 0
        
        for batch_num in range(len(embeddings) // batch_size):
            sel_labels = np.random.choice(unique_labels, 
                                        min(batch_size//8, len(unique_labels)), 
                                        replace=False)
            
            batch_idx, batch_labels = [], []
            for l in sel_labels:
                n = min(len(label_idx[l]), max(2, batch_size//len(sel_labels)))
                idx = np.random.choice(label_idx[l], n, replace=False)
                batch_idx.extend(idx)
                batch_labels.extend([l] * n)
            
            while len(batch_idx) < batch_size:
                extra_idx = np.random.randint(0, len(embeddings))
                batch_idx.append(extra_idx)
                batch_labels.append(labels[extra_idx])
            
            batch_idx = batch_idx[:batch_size]
            batch_labels = batch_labels[:batch_size]
            
            batch_emb = embeddings[batch_idx].to(device)
            
            projected = model(batch_emb)
            
            if torch.isnan(projected).any():
                print(f"❌ NaN in projected embeddings at epoch {epoch}, batch {batch_num}")
                continue
            
            loss = criterion(projected, batch_labels, sim_dict)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"❌ NaN/Inf loss at epoch {epoch}, batch {batch_num}")
                continue
            
            optimizer.zero_grad()
            loss.backward()
            
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                print(f"❌ NaN/Inf gradients at epoch {epoch}, batch {batch_num}")
                continue
            
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        
        if n_batches > 0:
            avg_loss = total_loss / n_batches
            
            if avg_loss < best_loss and not (torch.isnan(torch.tensor(avg_loss)) or torch.isinf(torch.tensor(avg_loss))):
                best_loss = avg_loss
                torch.save({'model': model.state_dict(), 'sim_dict': sim_dict}, 
                          'model/best_model.pth')
            
            if epoch % 5 == 0:
                print(f'Epoch {epoch}, Loss: {avg_loss:.4f}, Batches: {n_batches}')
        else:
            print(f'Epoch {epoch}: No valid batches')
    
    return model, sim_dict

def filter_outliers_by_label(proj_emb, labels, reports, item_ids, filter_strength=0.5):
    # 텐서를 numpy로 변환
    if torch.is_tensor(proj_emb):
        emb_np = proj_emb.cpu().numpy()
    else:
        emb_np = proj_emb.copy()
    
    print(f"\n=== OUTLIER FILTERING (strength={filter_strength}) ===")
    print(f"📊 Input projected embeddings: {emb_np.shape}")
    
    label_groups = defaultdict(list)
    for i, label in enumerate(labels):
        label_groups[label].append(i)
    
    keep_indices = []
    filter_stats = {}
    
    for label, indices in label_groups.items():
        if len(indices) < 3:
            keep_indices.extend(indices)
            filter_stats[label] = {
                'original': len(indices),
                'kept': len(indices), 
                'removed': 0,
                'reason': 'too_few_samples'
            }
            continue
        
        label_embs = emb_np[indices]
        centroid = label_embs.mean(axis=0)
        distances = cosine_distances([centroid], label_embs)[0]
        
        mean_dist = distances.mean()
        std_dist = distances.std()
        threshold = mean_dist + filter_strength * std_dist
        
        valid_mask = distances <= threshold
        valid_indices = [indices[i] for i in range(len(indices)) if valid_mask[i]]
        
        keep_indices.extend(valid_indices)
        
        filter_stats[label] = {
            'original': len(indices),
            'kept': len(valid_indices),
            'removed': len(indices) - len(valid_indices),
        }
    
    keep_indices = sorted(keep_indices)
    
    if torch.is_tensor(proj_emb):
        filtered_emb = proj_emb[keep_indices]
    else:
        filtered_emb = emb_np[keep_indices]
    
    filtered_reports = [reports[i] for i in keep_indices]
    filtered_labels = [labels[i] for i in keep_indices]
    filtered_ids = [item_ids[i] for i in keep_indices]
    
    total_removed = len(emb_np) - len(keep_indices)
    print(f"📊 Filtered embeddings: {len(filtered_emb)} (removed {total_removed})")
    print(f"   - Removal rate: {total_removed/len(emb_np)*100:.1f}%")
    
    return filtered_emb, filtered_reports, filtered_labels, filtered_ids

def build_db(proj_emb, reports, labels, item_ids, db_path="./medical_db", batch_size=5000):
    """ChromaDB 구축 - projected embeddings 저장"""
    if torch.is_tensor(proj_emb):
        proj_emb = proj_emb.cpu().numpy()
    
    print(f"\n=== BUILDING CHROMADB ===")
    print(f"📊 Projected embeddings to store: {proj_emb.shape}")
    
    import shutil
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
        print(f"Removed existing database: {db_path}")
    
    client = chromadb.PersistentClient(path=db_path)
    collection = client.create_collection("medical_images", 
                                        metadata={"hnsw:space": "cosine"})
    
    # Embedding augmentation mapping
    embedding_keys = [
        'macenko_original',
        'macenko_hsv', 
        'macenko_rgb_shift',
        'macenko_brightness_contrast',
        'macenko_he_aug'
    ]
    
    total_samples = len(proj_emb)
    print(f"Adding {total_samples} projected embeddings in batches of {batch_size}")
    
    for start_idx in range(0, total_samples, batch_size):
        end_idx = min(start_idx + batch_size, total_samples)
        
        print(f"Adding batch {start_idx//batch_size + 1}: samples {start_idx}-{end_idx-1}")
        
        batch_ids = []
        batch_metadatas = []
        
        for i in range(start_idx, end_idx):
            # Extract base filename and variation index from item_ids
            # item_ids format: "filename.tiff_v0", "filename.tiff_v1", etc.
            item_id = item_ids[i]
            
            if '_v' in item_id:
                base_filename, var_part = item_id.rsplit('_v', 1)
                var_idx = int(var_part)
                aug_type = embedding_keys[var_idx] if var_idx < len(embedding_keys) else f"variant_{var_idx}"
            else:
                base_filename = item_id
                aug_type = "original"
            
            # Remove .tiff extension from base_filename if present
            if base_filename.endswith('.tiff'):
                base_filename = base_filename[:-5]  # Remove .tiff
            
            # Create ID: base_filename + augmentation type
            db_id = f"{base_filename}_{aug_type}"
            batch_ids.append(db_id)
            
            batch_metadatas.append({
                "label": labels[i], 
                "item_id": base_filename,
                "augmentation": aug_type
            })
        
        collection.add(
            embeddings=proj_emb[start_idx:end_idx].tolist(),
            documents=reports[start_idx:end_idx],
            ids=batch_ids,
            metadatas=batch_metadatas
        )
    
    print(f"✅ Database built: {db_path}")
    print(f"   - Stored {total_samples} projected embeddings")
    print(f"   - ID format: filename_augmentation_type")
    return db_path



def main():
    # 1. 데이터 로딩
    print("=== STEP 1: LOADING DATA ===")
    valid_pairs = load_data()
    
    if not valid_pairs:
        print("❌ No valid pairs found")
        return
    
    # 2. 임베딩 추출
    print("\n=== STEP 2: EXTRACTING EMBEDDINGS ===")
    embeddings, reports, labels, item_ids = extract_embeddings(valid_pairs)
    
    if embeddings is None:
        print("❌ Embedding extraction failed")
        return
    
    # 3. 모델 학습
    print("\n=== STEP 3: TRAINING MODEL ===")
    model, sim_dict = train_model(embeddings, labels)
    
    if model is None:
        print("❌ Model training failed")
        return
    
    # 4. Projected embeddings 생성
    print("\n=== STEP 4: GENERATING PROJECTED EMBEDDINGS ===")
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        projected_embeddings = model(embeddings.to(device))
    
    print(f"✅ Generated projected embeddings: {projected_embeddings.shape}")
    print(f"   - Original: {embeddings.shape[1]}D -> Projected: {projected_embeddings.shape[1]}D")
    
    # 5. 아웃라이어 필터링
    print("\n=== STEP 5: FILTERING OUTLIERS ===")
    filtered_proj, filtered_reports, filtered_labels, filtered_ids = filter_outliers_by_label(
        projected_embeddings, labels, reports, item_ids)
    
    # 6. 모델 저장
    print("\n=== STEP 6: SAVING MODEL ===")
    os.makedirs('model', exist_ok=True)
    model_path = 'model/semantic_model.pth'
    torch.save({'model': model.state_dict(), 'sim_dict': sim_dict}, model_path)
    print(f"✅ Model saved: {model_path}")
    
    # 7. ChromaDB 구축 (projected embeddings 사용)
    print("\n=== STEP 7: BUILDING DATABASE ===")
    db_path = build_db(filtered_proj, filtered_reports, filtered_labels, filtered_ids)
    
    # 8. 최종 요약
    print(f"\n=== FINAL SUMMARY ===")
    print(f"📊 Data Flow:")
    print(f"   1. Original embeddings:     {embeddings.shape}")
    print(f"   2. Projected embeddings:    {projected_embeddings.shape}")
    print(f"   3. Filtered embeddings:     {len(filtered_proj)} × {filtered_proj.shape[1] if hasattr(filtered_proj, 'shape') else 'unknown'}")
    print(f"📁 Saved:")
    print(f"   - Model: {model_path}")
    print(f"   - Database: {db_path}")
    print(f"✅ Pipeline completed successfully!")

if __name__ == "__main__":
    main()