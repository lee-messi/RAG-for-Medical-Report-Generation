import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from collections import Counter
import chromadb
import torch
import os

def load_embeddings_from_db(db_path="./medical_db"):
    """ChromaDB에서 임베딩과 메타데이터 로드"""
    print(f"Loading embeddings from {db_path}")
    
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection("medical_images")
    
    results = collection.get(include=['embeddings', 'metadatas', 'documents'])
    
    embeddings = np.array(results['embeddings'])
    labels = [meta['label'] for meta in results['metadatas']]
    item_ids = [meta['item_id'] for meta in results['metadatas']]
    reports = results['documents']
    
    print(f"Loaded {len(embeddings)} embeddings")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Unique labels: {len(set(labels))}")
    
    return embeddings, labels, item_ids, reports

def visualize_3d_by_organs(emb_3d, labels, pca):
    """3D 장기별 임베딩 시각화"""
    print("\n=== 3D Visualizing by Organs ===")
    
    organ_labels = [label.split('_')[0] for label in labels]
    organ_counts = Counter(organ_labels)
    top_organs = sorted(organ_counts.keys())[:8]
    
    print(f"Top organs: {top_organs}")
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_organs)))
    
    for i, organ in enumerate(top_organs):
        mask = np.array(organ_labels) == organ
        ax.scatter(emb_3d[mask, 0], emb_3d[mask, 1], emb_3d[mask, 2],
                  c=[colors[i]], label=f"{organ} ({organ_counts[organ]})", 
                  alpha=0.7, s=20)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})')
    ax.set_title('3D Embeddings by Organ')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    os.makedirs('viz', exist_ok=True)
    filename = 'viz/embeddings_3d_by_organ.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {filename}")

def visualize_3d_by_labels(emb_3d, labels, pca):
    """3D 라벨별 임베딩 시각화 (상위 12개)"""
    print("\n=== 3D Visualizing by Labels (Top 12) ===")
    
    label_counts = Counter(labels)
    top_labels = [label for label, count in label_counts.most_common(12)]
    
    print("Top 12 labels:")
    for i, label in enumerate(top_labels):
        print(f"  {i+1:2d}. {label}: {label_counts[label]} samples")
    
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(top_labels)))
    
    for i, label in enumerate(top_labels):
        mask = np.array(labels) == label
        ax.scatter(emb_3d[mask, 0], emb_3d[mask, 1], emb_3d[mask, 2],
                  c=[colors[i]], label=f"{label[:12]}... ({label_counts[label]})", 
                  alpha=0.7, s=15)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})')
    ax.set_title('3D Embeddings by Label (Top 12)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    filename = 'viz/embeddings_3d_by_label.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {filename}")

def visualize_embeddings_3d_from_db(db_path="./medical_db"):
    """ChromaDB에서 임베딩을 로드하고 3D 시각화"""
    
    # 임베딩 로드
    embeddings, labels, item_ids, reports = load_embeddings_from_db(db_path)
    
    # 3D PCA 적용
    print(f"\n=== Applying 3D PCA ===")
    pca = PCA(n_components=3)
    emb_3d = pca.fit_transform(embeddings)
    
    print(f"Explained variance ratio:")
    print(f"  PC1: {pca.explained_variance_ratio_[0]:.1%}")
    print(f"  PC2: {pca.explained_variance_ratio_[1]:.1%}")
    print(f"  PC3: {pca.explained_variance_ratio_[2]:.1%}")
    print(f"  Total: {sum(pca.explained_variance_ratio_):.1%}")
    
    # 3D 시각화
    visualize_3d_by_organs(emb_3d, labels, pca)
    visualize_3d_by_labels(emb_3d, labels, pca)
    
    return embeddings, labels, item_ids, reports, emb_3d

# 실행
if __name__ == "__main__":
    embeddings, labels, item_ids, reports, emb_3d = visualize_embeddings_3d_from_db()
    print(f"\n3D visualization complete! Check the 'viz' folder for saved plots.")