import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import chromadb
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from collections import Counter
import warnings
import numpy as np
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

def extract_embeddings(valid_pairs, test_size=100, seed=1417):
    """임베딩 추출 - 각 NPZ에서 5개 variation"""
    np.random.seed(seed)
    
    # 임베딩 키들
    embedding_keys = [
        'embedding_macenko_original',
        'embedding_macenko_hsv', 
        'embedding_macenko_rgb_shift',
        'embedding_macenko_brightness_contrast',
        'embedding_macenko_he_aug'
    ]
    
    indices = np.random.permutation(len(valid_pairs))
    test_size = min(test_size, len(valid_pairs) // 5)
    train_indices = indices[test_size:]
    
    # Test 파일명 저장
    test_filenames = [valid_pairs[i][0]['id'] for i in indices[:test_size]]
    with open('test_filenames.txt', 'w') as f:
        for filename in test_filenames:
            f.write(f"{filename}\n")
    
    embeddings, reports, labels, item_ids = [], [], [], []
    
    for i in train_indices:
        item, npz_path = valid_pairs[i]
        
        try:
            data = np.load(npz_path, allow_pickle=True)
            
            # 5개 임베딩 키에서 각각 추출
            for j, key in enumerate(embedding_keys):
                if key in data:
                    emb = data[key]
                    # 1D 배열인 경우 그대로 사용
                    if len(emb.shape) == 1:
                        embeddings.append(emb)
                    # 2D 배열인 경우 첫 번째 행 사용
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
    print(f"Extracted {len(embeddings)} embeddings from {len(train_indices)} files")
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
    
    if tokens1[0] != tokens2[0]:  # 다른 장기
        return 0.0
    
    # histologic type tokens (exclude organ and procedure)
    hist_tokens1 = tokens1[2:]
    hist_tokens2 = tokens2[2:]
    
    # Same histologic type tokens
    if hist_tokens1 == hist_tokens2:
        bleu_score = 1.0
    # Not enough tokens for BLEU-4
    elif len(hist_tokens1) < 4 or len(hist_tokens2) < 4:
        bleu_score = 0.0
    else:
        # Enough tokens - use BLEU-4
        hist_type1 = ' '.join(hist_tokens1)
        hist_type2 = ' '.join(hist_tokens2)
        bleu_score = get_bleu4(hist_type1, hist_type2)
    
    # Always calculate ROUGE
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

def compare_embeddings_by_organs(original_emb, projected_emb, labels, projected_labels=None):
    """장기별 임베딩 비교 (원본 vs 투영)"""
    print(f"\n=== ORGAN COMPARISON: ORIGINAL vs PROJECTED ===")
    
    # projected_labels가 제공되지 않으면 labels 사용 (기존 호출 방식 지원)
    if projected_labels is None:
        projected_labels = labels
    
    # 텐서를 numpy로 변환
    if torch.is_tensor(original_emb):
        orig_np = original_emb.cpu().numpy()
    else:
        orig_np = original_emb
        
    if torch.is_tensor(projected_emb):
        proj_np = projected_emb.cpu().numpy()
    else:
        proj_np = projected_emb
    
    # 장기 라벨 생성 (각각 다른 labels 사용)
    orig_organ_labels = [label.split('_')[0] for label in labels]
    proj_organ_labels = [label.split('_')[0] for label in projected_labels]
    
    # 공통 장기들만 선택
    orig_organ_counts = Counter(orig_organ_labels)
    proj_organ_counts = Counter(proj_organ_labels)
    common_organs = set(orig_organ_counts.keys()) & set(proj_organ_counts.keys())
    top_organs = sorted(list(common_organs))[:8]
    
    # PCA 적용
    pca_orig = PCA(n_components=2)
    pca_proj = PCA(n_components=2)
    
    orig_2d = pca_orig.fit_transform(orig_np)
    proj_2d = pca_proj.fit_transform(proj_np)
    
    # 비교 플롯
    plt.figure(figsize=(16, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_organs)))
    
    # 원본 임베딩
    plt.subplot(1, 2, 1)
    for i, organ in enumerate(top_organs):
        mask = np.array(orig_organ_labels) == organ  # labels 기반
        plt.scatter(orig_2d[mask, 0], orig_2d[mask, 1], 
                   c=[colors[i]], label=f"{organ} ({orig_organ_counts[organ]})", 
                   alpha=0.7, s=20)
    
    plt.xlabel(f'PC1 ({pca_orig.explained_variance_ratio_[0]:.1%})')
    plt.ylabel(f'PC2 ({pca_orig.explained_variance_ratio_[1]:.1%})')
    plt.title('Original Embeddings - By Organ')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # 투영된 임베딩
    plt.subplot(1, 2, 2)
    for i, organ in enumerate(top_organs):
        mask = np.array(proj_organ_labels) == organ  # projected_labels 기반
        plt.scatter(proj_2d[mask, 0], proj_2d[mask, 1], 
                   c=[colors[i]], label=f"{organ} ({proj_organ_counts[organ]})", 
                   alpha=0.7, s=20)
    
    plt.xlabel(f'PC1 ({pca_proj.explained_variance_ratio_[0]:.1%})')
    plt.ylabel(f'PC2 ({pca_proj.explained_variance_ratio_[1]:.1%})')
    plt.title('Projected Embeddings - By Organ')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = 'viz/organ_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Saved organ comparison: {filename}")
    return filename

def compare_embeddings_by_labels(original_emb, projected_emb, labels, projected_labels=None):
    """상위 20개 라벨별 임베딩 비교 (원본 vs 투영)"""
    print(f"\n=== LABEL COMPARISON: ORIGINAL vs PROJECTED (Top 20) ===")
    
    if projected_labels is None:
        projected_labels = labels
    
    print(f"Dataset sizes: Original={len(labels)}, Projected={len(projected_labels)}")
    
    # 텐서를 numpy로 변환
    if torch.is_tensor(original_emb):
        orig_np = original_emb.cpu().numpy()
    else:
        orig_np = original_emb
        
    if torch.is_tensor(projected_emb):
        proj_np = projected_emb.cpu().numpy()
    else:
        proj_np = projected_emb
    
    # 공통 라벨들 찾기
    orig_counts = Counter(labels)
    proj_counts = Counter(projected_labels)
    common_labels = set(orig_counts.keys()) & set(proj_counts.keys())
    
    print(f"Unique labels: Original={len(orig_counts)}, Projected={len(proj_counts)}, Common={len(common_labels)}")
    
    # 최소 샘플 수를 고정값으로 설정 (기존 계산이 너무 보수적)
    min_samples = 5  # 고정값 사용
    
    # 두 데이터셋 모두에서 최소 샘플 수를 만족하는 라벨들
    valid_labels = [label for label in common_labels 
                   if orig_counts[label] >= min_samples and proj_counts[label] >= min_samples]
    
    # 두 데이터셋의 평균 빈도로 정렬
    label_avg_counts = [(label, (orig_counts[label] + proj_counts[label]) / 2) 
                       for label in valid_labels]
    label_avg_counts.sort(key=lambda x: x[1], reverse=True)
    top_labels = [label for label, _ in label_avg_counts[:20]]
    
    print(f"Labels with sufficient samples in both datasets (min={min_samples}):")
    print(f"Total valid labels: {len(valid_labels)}, showing top {len(top_labels)}")
    for i, label in enumerate(top_labels):
        orig_count = orig_counts[label]
        proj_count = proj_counts[label]
        avg_count = (orig_count + proj_count) / 2
        print(f"  {i+1:2d}. {label}: orig={orig_count}, proj={proj_count}, avg={avg_count:.1f}")
    
    # PCA 적용
    pca_orig = PCA(n_components=2)
    pca_proj = PCA(n_components=2)
    
    orig_2d = pca_orig.fit_transform(orig_np)
    proj_2d = pca_proj.fit_transform(proj_np)
    
    # 비교 플롯
    plt.figure(figsize=(20, 8))
    colors = plt.cm.tab20(np.linspace(0, 1, len(top_labels)))
    
    # 원본 임베딩
    plt.subplot(1, 2, 1)
    for i, label in enumerate(top_labels):
        mask = np.array(labels) == label
        if mask.sum() > 0:
            plt.scatter(orig_2d[mask, 0], orig_2d[mask, 1], 
                       c=[colors[i]], label=f"{label[:12]}... ({orig_counts[label]})", 
                       alpha=0.7, s=15)
    
    plt.xlabel(f'PC1 ({pca_orig.explained_variance_ratio_[0]:.1%})')
    plt.ylabel(f'PC2 ({pca_orig.explained_variance_ratio_[1]:.1%})')
    plt.title('Original Embeddings - By Label (Top 20)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
    plt.grid(True, alpha=0.3)
    
    # 투영된 임베딩
    plt.subplot(1, 2, 2)
    for i, label in enumerate(top_labels):
        mask = np.array(projected_labels) == label
        if mask.sum() > 0:
            plt.scatter(proj_2d[mask, 0], proj_2d[mask, 1], 
                       c=[colors[i]], label=f"{label[:12]}... ({proj_counts[label]})", 
                       alpha=0.7, s=15)
    
    plt.xlabel(f'PC1 ({pca_proj.explained_variance_ratio_[0]:.1%})')
    plt.ylabel(f'PC2 ({pca_proj.explained_variance_ratio_[1]:.1%})')
    plt.title('Projected Embeddings - By Label (Top 20)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = 'viz/label_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Saved label comparison: {filename}")
    return filename

def visualize_similarity_matrix(labels, sample_size=10):
    """유사도 매트릭스 시각화"""
    print(f"\n=== SIMILARITY MATRIX VISUALIZATION ===")
    
    # 샘플 라벨 선택
    unique_labels = list(set(labels))
    if len(unique_labels) > sample_size:
        sample_labels = np.random.choice(unique_labels, sample_size, replace=False)
    else:
        sample_labels = unique_labels
    
    print(f"Sample labels ({len(sample_labels)}):")
    for i, label in enumerate(sample_labels):
        print(f"  {i}: {label}")
    
    # 유사도 매트릭스 계산
    matrix = np.zeros((len(sample_labels), len(sample_labels)))
    
    for i, l1 in enumerate(sample_labels):
        for j, l2 in enumerate(sample_labels):
            matrix[i, j] = compute_label_similarity(l1, l2)
    
    # 텍스트로 출력
    print(f"\nSimilarity Matrix ({len(sample_labels)}x{len(sample_labels)}):")
    print("     ", end="")
    for i in range(len(sample_labels)):
        print(f"{i:>6}", end="")
    print()
    
    for i in range(len(sample_labels)):
        print(f"{i:>3}: ", end="")
        for j in range(len(sample_labels)):
            print(f"{matrix[i,j]:>6.2f}", end="")
        print()
    
    # 시각적 매트릭스 (matplotlib)
    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(label='Similarity Score')
    
    # 값 표시
    for i in range(len(sample_labels)):
        for j in range(len(sample_labels)):
            plt.text(j, i, f'{matrix[i,j]:.2f}', 
                    ha='center', va='center', 
                    color='white' if matrix[i,j] > 0.5 else 'black')
    
    # 축 라벨 (짧게)
    short_labels = [label.split('_')[0] + '_' + label.split('_')[1] if '_' in label else label 
                   for label in sample_labels]
    
    plt.xticks(range(len(sample_labels)), short_labels, rotation=45, ha='right')
    plt.yticks(range(len(sample_labels)), short_labels)
    plt.xlabel('Labels')
    plt.ylabel('Labels')
    plt.title('Label Similarity Matrix (Sample)')
    plt.tight_layout()
    
    # plt.savefig('similarity_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 통계 정보
    print(f"\nMatrix Statistics:")
    print(f"  Min similarity: {matrix.min():.3f}")
    print(f"  Max similarity: {matrix.max():.3f}")
    print(f"  Mean similarity: {matrix.mean():.3f}")
    print(f"  Diagonal (self-similarity): all = 1.0")
    
    # 유사도 분포
    off_diagonal = matrix[np.triu_indices_from(matrix, k=1)]
    print(f"  Off-diagonal similarities:")
    print(f"    0.0 (different organs): {(off_diagonal == 0.0).sum()}")
    print(f"    >0.0 and <1.0 (same organ, different pathology): {((off_diagonal > 0.0) & (off_diagonal < 1.0)).sum()}")
    print(f"    1.0 (identical): {(off_diagonal == 1.0).sum()}")
    
    return matrix, sample_labels

class SemanticContrastiveLoss(nn.Module):
    def __init__(self, temp=0.07, organ_weight=2.0, fine_weight=4.0):
        super().__init__()
        self.temp = max(temp, 1e-8)  # 최소값 설정
        self.organ_w = organ_weight
        self.fine_w = fine_weight
    
    def forward(self, features, batch_labels, sim_dict):
        bs = features.shape[0]
        device = features.device
        
        # NaN/Inf 체크
        if torch.isnan(features).any() or torch.isinf(features).any():
            print("❌ NaN/Inf in features")
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # 유사도 매트릭스
        sim_matrix = torch.zeros(bs, bs, device=device)
        for i in range(bs):
            for j in range(bs):
                sim_matrix[i, j] = sim_dict.get((batch_labels[i], batch_labels[j]), 0.0)
        
        # 코사인 유사도 - features가 이미 normalized되어 있음
        cosine_sim = torch.matmul(features, features.T) / self.temp
        
        # NaN/Inf 체크
        if torch.isnan(cosine_sim).any() or torch.isinf(cosine_sim).any():
            print("❌ NaN/Inf in cosine_sim")
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # 대각선 마스킹
        mask = torch.eye(bs, dtype=torch.bool, device=device)
        cosine_sim = cosine_sim.masked_fill(mask, -1e9)  # -inf 대신 큰 음수
        
        # Positive 마스크
        organ_pos = (sim_matrix > 0.0) & (sim_matrix < 1.0)
        exact_pos = (sim_matrix == 1.0)
        
        organ_pos = organ_pos.float().masked_fill(mask, 0)
        exact_pos = exact_pos.float().masked_fill(mask, 0)
        
        has_pos = ((organ_pos + exact_pos).sum(1) > 0)
        if has_pos.sum() == 0:
            print("⚠️ No positive pairs found")
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Numerically stable softmax
        logits_max = torch.max(cosine_sim, dim=1, keepdim=True)[0]
        logits = cosine_sim - logits_max.detach()
        
        # Clamp to prevent overflow
        logits = torch.clamp(logits, min=-50, max=50)
        
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        
        # NaN 체크
        if torch.isnan(log_prob).any():
            print("❌ NaN in log_prob")
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Weighted positive
        weighted_pos = (organ_pos * sim_matrix * self.fine_w + exact_pos * self.organ_w)
        
        # 0으로 나누기 방지
        pos_sum = weighted_pos.sum(1)
        pos_sum = torch.clamp(pos_sum, min=1e-8)
        
        mean_log_prob = (weighted_pos * log_prob).sum(1) / pos_sum
        
        loss = -mean_log_prob[has_pos].mean()
        
        # 최종 NaN 체크
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
    
    # 입력 데이터 검증
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embeddings stats: min={embeddings.min():.4f}, max={embeddings.max():.4f}, mean={embeddings.mean():.4f}")
    
    if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
        print("❌ NaN/Inf in input embeddings")
        return None, None
    
    sim_dict = create_similarity_matrix(labels)
    print(f"Created similarity matrix with {len(sim_dict)} pairs")
    
    model = ProjectionHead(embeddings.shape[1]).to(device)
    criterion = SemanticContrastiveLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)  # 더 작은 lr
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    unique_labels = list(set(labels))
    label_idx = {l: [i for i, x in enumerate(labels) if x == l] for l in unique_labels}
    
    print(f"Unique labels: {len(unique_labels)}")
    print(f"Label distribution: {[(l, len(label_idx[l])) for l in list(unique_labels)[:5]]}")
    
    model.train()
    best_loss = float('inf')
    
    for epoch in range(epochs):
        total_loss, n_batches = 0, 0
        
        for batch_num in range(len(embeddings) // batch_size):
            # 배치 구성 - 더 안전하게
            sel_labels = np.random.choice(unique_labels, 
                                        min(batch_size//8, len(unique_labels)), 
                                        replace=False)
            
            batch_idx, batch_labels = [], []
            for l in sel_labels:
                n = min(len(label_idx[l]), max(2, batch_size//len(sel_labels)))  # 최소 2개
                idx = np.random.choice(label_idx[l], n, replace=False)
                batch_idx.extend(idx)
                batch_labels.extend([l] * n)
            
            # 배치 크기 맞추기
            while len(batch_idx) < batch_size:
                extra_idx = np.random.randint(0, len(embeddings))
                batch_idx.append(extra_idx)
                batch_labels.append(labels[extra_idx])
            
            batch_idx = batch_idx[:batch_size]
            batch_labels = batch_labels[:batch_size]
            
            batch_emb = embeddings[batch_idx].to(device)
            
            # Forward pass
            projected = model(batch_emb)
            
            # NaN 체크
            if torch.isnan(projected).any():
                print(f"❌ NaN in projected embeddings at epoch {epoch}, batch {batch_num}")
                continue
            
            loss = criterion(projected, batch_labels, sim_dict)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"❌ NaN/Inf loss at epoch {epoch}, batch {batch_num}")
                continue
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            
            # Gradient 체크
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

def build_db(proj_emb, reports, labels, item_ids, db_path="./medical_db", batch_size=5000):
    """ChromaDB 구축 (배치 처리) - 기존 DB 완전 교체"""
    if torch.is_tensor(proj_emb):
        proj_emb = proj_emb.cpu().numpy()
    
    # 기존 DB 디렉토리 완전 삭제
    import shutil
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
        print(f"Removed existing database: {db_path}")
    
    client = chromadb.PersistentClient(path=db_path)
    collection = client.create_collection("medical_images", 
                                        metadata={"hnsw:space": "cosine"})
    
    total_samples = len(proj_emb)
    print(f"Adding {total_samples} embeddings in batches of {batch_size}")
    
    # 배치 단위로 추가
    for start_idx in range(0, total_samples, batch_size):
        end_idx = min(start_idx + batch_size, total_samples)
        
        print(f"Adding batch {start_idx//batch_size + 1}: samples {start_idx}-{end_idx-1}")
        
        collection.add(
            embeddings=proj_emb[start_idx:end_idx].tolist(),
            documents=reports[start_idx:end_idx],
            ids=[f"{item_ids[i]}_{i}" for i in range(start_idx, end_idx)],
            metadatas=[{"label": labels[i], "item_id": str(item_ids[i])} 
                      for i in range(start_idx, end_idx)]
        )
    
    print(f"Database built: {db_path}")
    print(f"Stored {total_samples} embeddings with labels")
    return db_path

def compare_embeddings_by_organs(original_emb, projected_emb, labels, projected_labels=None):
    """장기별 임베딩 비교 (원본 vs 투영)"""
    print(f"\n=== ORGAN COMPARISON: ORIGINAL vs PROJECTED ===")
    
    # projected_labels가 제공되지 않으면 labels 사용 (기존 호출 방식 지원)
    if projected_labels is None:
        projected_labels = labels
    
    # 텐서를 numpy로 변환
    if torch.is_tensor(original_emb):
        orig_np = original_emb.cpu().numpy()
    else:
        orig_np = original_emb
        
    if torch.is_tensor(projected_emb):
        proj_np = projected_emb.cpu().numpy()
    else:
        proj_np = projected_emb
    
    # 장기 라벨 생성 (각각 다른 labels 사용)
    orig_organ_labels = [label.split('_')[0] for label in labels]
    proj_organ_labels = [label.split('_')[0] for label in projected_labels]
    
    # 공통 장기들만 선택
    orig_organ_counts = Counter(orig_organ_labels)
    proj_organ_counts = Counter(proj_organ_labels)
    common_organs = set(orig_organ_counts.keys()) & set(proj_organ_counts.keys())
    top_organs = sorted(list(common_organs))[:8]
    
    # PCA 적용
    pca_orig = PCA(n_components=2)
    pca_proj = PCA(n_components=2)
    
    orig_2d = pca_orig.fit_transform(orig_np)
    proj_2d = pca_proj.fit_transform(proj_np)
    
    # 비교 플롯
    plt.figure(figsize=(16, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_organs)))
    
    # 원본 임베딩
    plt.subplot(1, 2, 1)
    for i, organ in enumerate(top_organs):
        mask = np.array(orig_organ_labels) == organ  # labels 기반
        plt.scatter(orig_2d[mask, 0], orig_2d[mask, 1], 
                   c=[colors[i]], label=f"{organ} ({orig_organ_counts[organ]})", 
                   alpha=0.7, s=20)
    
    plt.xlabel(f'PC1 ({pca_orig.explained_variance_ratio_[0]:.1%})')
    plt.ylabel(f'PC2 ({pca_orig.explained_variance_ratio_[1]:.1%})')
    plt.title('Original Embeddings - By Organ')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # 투영된 임베딩
    plt.subplot(1, 2, 2)
    for i, organ in enumerate(top_organs):
        mask = np.array(proj_organ_labels) == organ  # projected_labels 기반
        plt.scatter(proj_2d[mask, 0], proj_2d[mask, 1], 
                   c=[colors[i]], label=f"{organ} ({proj_organ_counts[organ]})", 
                   alpha=0.7, s=20)
    
    plt.xlabel(f'PC1 ({pca_proj.explained_variance_ratio_[0]:.1%})')
    plt.ylabel(f'PC2 ({pca_proj.explained_variance_ratio_[1]:.1%})')
    plt.title('Projected Embeddings - By Organ')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = 'viz/organ_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Saved organ comparison: {filename}")
    return filename

def compare_embeddings_by_labels(original_emb, projected_emb, labels, projected_labels=None):
    """상위 20개 라벨별 임베딩 비교 (원본 vs 투영)"""
    print(f"\n=== LABEL COMPARISON: ORIGINAL vs PROJECTED (Top 20) ===")
    
    # projected_labels가 제공되지 않으면 labels 사용 (기존 호출 방식 지원)
    if projected_labels is None:
        projected_labels = labels
    
    # 텐서를 numpy로 변환
    if torch.is_tensor(original_emb):
        orig_np = original_emb.cpu().numpy()
    else:
        orig_np = original_emb
        
    if torch.is_tensor(projected_emb):
        proj_np = projected_emb.cpu().numpy()
    else:
        proj_np = projected_emb
    
    # 공통 라벨들 찾기
    orig_counts = Counter(labels)
    proj_counts = Counter(projected_labels)
    common_labels = set(orig_counts.keys()) & set(proj_counts.keys())
    
    # 공통 라벨 중 상위 20개 (원본 기준)
    top_labels = sorted([label for label in common_labels 
                        if orig_counts[label] >= 2])[:20]  # 최소 2개 이상
    
    print(f"Top labels (common in both):")
    for i, label in enumerate(top_labels):
        orig_count = orig_counts[label]
        proj_count = proj_counts[label]
        print(f"  {i+1:2d}. {label}: orig={orig_count}, proj={proj_count}")
    
    # PCA 적용
    pca_orig = PCA(n_components=2)
    pca_proj = PCA(n_components=2)
    
    orig_2d = pca_orig.fit_transform(orig_np)
    proj_2d = pca_proj.fit_transform(proj_np)
    
    # 비교 플롯
    plt.figure(figsize=(20, 8))
    colors = plt.cm.tab20(np.linspace(0, 1, len(top_labels)))
    
    # 원본 임베딩
    plt.subplot(1, 2, 1)
    for i, label in enumerate(top_labels):
        mask = np.array(labels) == label  # labels 기반
        if mask.sum() > 0:  # 해당 라벨이 있는지 확인
            plt.scatter(orig_2d[mask, 0], orig_2d[mask, 1], 
                       c=[colors[i]], label=f"{label[:12]}... ({orig_counts[label]})", 
                       alpha=0.7, s=15)
    
    plt.xlabel(f'PC1 ({pca_orig.explained_variance_ratio_[0]:.1%})')
    plt.ylabel(f'PC2 ({pca_orig.explained_variance_ratio_[1]:.1%})')
    plt.title('Original Embeddings - By Label (Top 20)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
    plt.grid(True, alpha=0.3)
    
    # 투영된 임베딩
    plt.subplot(1, 2, 2)
    for i, label in enumerate(top_labels):
        mask = np.array(projected_labels) == label  # projected_labels 기반
        if mask.sum() > 0:  # 해당 라벨이 있는지 확인
            plt.scatter(proj_2d[mask, 0], proj_2d[mask, 1], 
                       c=[colors[i]], label=f"{label[:12]}... ({proj_counts[label]})", 
                       alpha=0.7, s=15)
    
    plt.xlabel(f'PC1 ({pca_proj.explained_variance_ratio_[0]:.1%})')
    plt.ylabel(f'PC2 ({pca_proj.explained_variance_ratio_[1]:.1%})')
    plt.title('Projected Embeddings - By Label (Top 20)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = 'viz/label_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Saved label comparison: {filename}")
    return filename

def filter_outliers_by_label(proj_emb, labels, reports, item_ids, filter_strength=0.5):
    
    # 텐서를 numpy로 변환
    if torch.is_tensor(proj_emb):
        emb_np = proj_emb.cpu().numpy()
    else:
        emb_np = proj_emb.copy()
    
    print(f"\n=== OUTLIER FILTERING (strength={filter_strength}) ===")
    print(f"Original: {len(emb_np)} samples")
    
    # 라벨별 그룹핑
    label_groups = defaultdict(list)
    for i, label in enumerate(labels):
        label_groups[label].append(i)
    
    keep_indices = []
    filter_stats = {}
    
    for label, indices in label_groups.items():
        if len(indices) < 3:  # 샘플이 너무 적으면 필터링 안함
            keep_indices.extend(indices)
            filter_stats[label] = {
                'original': len(indices),
                'kept': len(indices), 
                'removed': 0,
                'threshold': 0.0,
                'reason': 'too_few_samples'
            }
            continue
        
        # 해당 라벨의 임베딩들
        label_embs = emb_np[indices]
        
        # 중심점 계산 (평균)
        centroid = label_embs.mean(axis=0)
        
        # 코사인 거리 계산 (1 - cosine_similarity)
        distances = cosine_distances([centroid], label_embs)[0]
        
        # 임계값 계산: mean + filter_strength * std
        mean_dist = distances.mean()
        std_dist = distances.std()
        threshold = mean_dist + filter_strength * std_dist
        
        # 필터링
        valid_mask = distances <= threshold
        valid_indices = [indices[i] for i in range(len(indices)) if valid_mask[i]]
        
        keep_indices.extend(valid_indices)
        
        filter_stats[label] = {
            'original': len(indices),
            'kept': len(valid_indices),
            'removed': len(indices) - len(valid_indices),
            'threshold': threshold,
            'mean_dist': mean_dist,
            'std_dist': std_dist,
            'max_dist': distances.max(),
            'min_dist': distances.min()
        }
    
    # 필터링된 데이터 생성
    keep_indices = sorted(keep_indices)
    
    if torch.is_tensor(proj_emb):
        filtered_emb = proj_emb[keep_indices]
    else:
        filtered_emb = emb_np[keep_indices]
    
    filtered_reports = [reports[i] for i in keep_indices]
    filtered_labels = [labels[i] for i in keep_indices]
    filtered_ids = [item_ids[i] for i in keep_indices]
    
    # 통계 출력
    total_removed = len(emb_np) - len(keep_indices)
    print(f"Filtered: {len(filtered_emb)} samples (removed {total_removed})")
    print(f"Removal rate: {total_removed/len(emb_np)*100:.1f}%")
    
    print(f"\nPer-label statistics:")
    removed_by_label = []
    for label, stats in sorted(filter_stats.items(), key=lambda x: x[1]['removed'], reverse=True):
        if stats['removed'] > 0:
            removal_rate = stats['removed'] / stats['original'] * 100
            print(f"  {label[:30]:30s}: {stats['removed']:3d}/{stats['original']:3d} removed ({removal_rate:5.1f}%)")
            removed_by_label.append((label, stats['removed'], removal_rate))
    
    if len(removed_by_label) > 0:
        print(f"\nTop labels with most removals:")
        for i, (label, removed, rate) in enumerate(removed_by_label[:5]):
            print(f"  {i+1}. {label}: {removed} samples ({rate:.1f}%)")
    else:
        print("No samples were removed.")
    
    return filtered_emb, filtered_reports, filtered_labels, filtered_ids

def main():
    # 데이터 로딩
    valid_pairs = load_data()
    
    if not valid_pairs:
        print("No valid pairs found")
        return
    
    # 임베딩 추출
    embeddings, reports, labels, item_ids = extract_embeddings(valid_pairs)
    
    if embeddings is None:
        print("Embedding extraction failed")
        return
    
    # 유사도 매트릭스 시각화
    # matrix, sample_labels = visualize_similarity_matrix(labels, sample_size=10)
    
    # 모델 학습
    print("Training model...")
    model, sim_dict = train_model(embeddings, labels)
    
    # 변환
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        transformed = model(embeddings.to(device))

    # 필터링
    filtered, filtered_reports, filtered_labels, filtered_ids = filter_outliers_by_label(
        transformed, labels, reports, item_ids)

    # 모델 저장
    print(f"  Model: semantic_model.pth")
    torch.save({'model': model.state_dict(), 'sim_dict': sim_dict}, 
              'model/semantic_model.pth')

    # DB 구축
    save_chroma_db = False
    if save_chroma_db: 
        db_path = build_db(filtered, filtered_reports, filtered_labels, filtered_ids)
        print(f"  DB: {db_path}")

if __name__ == "__main__":
    main()