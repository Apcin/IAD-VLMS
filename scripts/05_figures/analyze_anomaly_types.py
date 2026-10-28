#!/usr/bin/env python3
"""
Analyze Anomaly Types Distribution using Nomic Embed v2 MoE
============================================================

This script:
1. Extracts all <type> tags from the unified dataset
2. Embeds them using nomic-embed-text-v2-moe
3. Computes pairwise semantic distances
4. Performs clustering and visualization
5. Provides statistics to calibrate reward scales

Usage:
    python analyze_anomaly_types.py --dataset unified_train_oneshot_full.json
"""

import json
import re
import os
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Tuple
import numpy as np
import torch
from tqdm import tqdm

# Set HF cache
os.environ["HF_HOME"] = (os.environ.get('WORK_DIR','/bulk/aacudad/reasoning_traces')+"/hf_cache")

print("Loading dependencies...")
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# Nomic Embedding Model
# ==============================================================================

class NomicEmbedModel:
    """Nomic Embed v2 MoE for semantic embeddings."""
    
    def __init__(self, device="cpu"):
        self.device = device
        print(f"Loading nomic-embed-text-v2-moe on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "nomic-ai/nomic-embed-text-v2-moe",
            trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            "nomic-ai/nomic-embed-text-v2-moe",
            trust_remote_code=True,
            torch_dtype=torch.float32
        ).to(device)
        self.model.eval()
        print("Model loaded!")
    
    def embed(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Embed a list of texts."""
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            batch = texts[i:i + batch_size]
            # Add prefix for document embedding
            batch = [f"search_document: {text}" for text in batch]
            
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_embeddings = outputs.last_hidden_state.mean(dim=1)
                embeddings.append(batch_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute pairwise cosine similarity matrix."""
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / norms
        
        # Compute cosine similarity
        similarity = normalized @ normalized.T
        return similarity

# ==============================================================================
# Data Extraction
# ==============================================================================

def extract_types_from_dataset(json_path: str) -> Tuple[List[str], List[Dict]]:
    """
    Extract all <type> tags from dataset.
    
    Returns:
        types: List of extracted type strings
        metadata: List of dicts with context (product, source, etc.)
    """
    print(f"\nLoading dataset from: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print(f"Found {len(data)} samples")
    
    types = []
    metadata = []
    
    for idx, entry in enumerate(tqdm(data, desc="Extracting types")):
        # Try to extract from original_reasoning
        text = entry.get('original_reasoning', '')
        
        # If not there, try conversation
        if not text:
            convs = entry.get('conversation', [])
            if convs:
                text = convs[0].get('Answer', '')
        
        # Extract <type>...</type>
        type_match = re.search(r'<type>(.*?)</type>', text, re.IGNORECASE | re.DOTALL)
        
        if type_match:
            type_str = type_match.group(1).strip()
            if type_str:  # Non-empty
                types.append(type_str)
                metadata.append({
                    'index': idx,
                    'product': entry.get('product', 'unknown'),
                    'image_id': entry.get('image_id', ''),
                    'type': type_str
                })
    
    print(f"\nExtracted {len(types)} anomaly types from {len(data)} samples")
    print(f"Unique types: {len(set(types))}")
    
    return types, metadata

# ==============================================================================
# Analysis Functions
# ==============================================================================

def analyze_type_distribution(types: List[str], metadata: List[Dict]):
    """Analyze the distribution of types."""
    print("\n" + "="*80)
    print("TYPE DISTRIBUTION ANALYSIS")
    print("="*80)
    
    # Count occurrences
    type_counts = Counter(types)
    
    print(f"\nTotal types: {len(types)}")
    print(f"Unique types: {len(type_counts)}")
    print(f"\nTop 20 most common types:")
    for type_str, count in type_counts.most_common(20):
        print(f"  {count:5d}x  {type_str[:60]}")
    
    # Analyze by product
    product_types = defaultdict(set)
    for meta in metadata:
        product_types[meta['product']].add(meta['type'])
    
    print(f"\nTypes per product:")
    for product, ptypes in sorted(product_types.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
        print(f"  {product:20s}: {len(ptypes)} unique types")
    
    return type_counts

def compute_distance_statistics(embeddings: np.ndarray, types: List[str]):
    """Compute pairwise distance statistics."""
    print("\n" + "="*80)
    print("SEMANTIC DISTANCE ANALYSIS")
    print("="*80)
    
    # Compute similarity matrix
    print("\nComputing pairwise similarities...")
    similarity = embeddings @ embeddings.T / (
        np.linalg.norm(embeddings, axis=1)[:, None] * 
        np.linalg.norm(embeddings, axis=1)[None, :]
    )
    
    # Convert to distance (1 - similarity)
    distance = 1 - similarity
    
    # Get upper triangle (no self-comparisons)
    triu_indices = np.triu_indices_from(distance, k=1)
    distances = distance[triu_indices]
    
    print(f"\nDistance Statistics (1 - cosine_similarity):")
    print(f"  Min distance:  {distances.min():.4f}")
    print(f"  Max distance:  {distances.max():.4f}")
    print(f"  Mean distance: {distances.mean():.4f}")
    print(f"  Std distance:  {distances.std():.4f}")
    print(f"  Median:        {np.median(distances):.4f}")
    
    print(f"\nPercentiles:")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        print(f"  {p:2d}th: {np.percentile(distances, p):.4f}")
    
    # Find most similar pairs
    print(f"\n10 Most Similar Type Pairs:")
    flat_indices = np.argsort(distances)[:10]
    for rank, idx in enumerate(flat_indices, 1):
        i, j = triu_indices[0][idx], triu_indices[1][idx]
        sim = 1 - distances[idx]
        print(f"  {rank}. {sim:.4f}  |  {types[i][:30]:30s} <-> {types[j][:30]:30s}")
    
    # Find most dissimilar pairs
    print(f"\n10 Most Dissimilar Type Pairs:")
    flat_indices = np.argsort(distances)[-10:][::-1]
    for rank, idx in enumerate(flat_indices, 1):
        i, j = triu_indices[0][idx], triu_indices[1][idx]
        sim = 1 - distances[idx]
        print(f"  {rank}. {sim:.4f}  |  {types[i][:30]:30s} <-> {types[j][:30]:30s}")
    
    return distance, similarity

def perform_clustering(embeddings: np.ndarray, types: List[str], n_clusters: int = 10):
    """Perform K-means clustering."""
    print("\n" + "="*80)
    print(f"K-MEANS CLUSTERING (k={n_clusters})")
    print("="*80)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    
    # Analyze clusters
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_types = [types[i] for i in cluster_indices]
        
        print(f"\nCluster {cluster_id} ({len(cluster_indices)} types):")
        # Show most common types in this cluster
        cluster_counter = Counter(cluster_types)
        for type_str, count in cluster_counter.most_common(5):
            print(f"  {count:4d}x  {type_str[:60]}")
    
    return labels

def visualize_embeddings(embeddings: np.ndarray, types: List[str], labels: np.ndarray, 
                        output_dir: str = "anomaly_type_analysis"):
    """Visualize embeddings using t-SNE."""
    print("\n" + "="*80)
    print("VISUALIZATION")
    print("="*80)
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # t-SNE reduction
    print("\nRunning t-SNE (this may take a while)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plot 1: Colored by cluster
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=labels, cmap='tab10', alpha=0.6, s=50)
    plt.colorbar(scatter, label='Cluster')
    plt.title('Anomaly Types - t-SNE Visualization (Colored by Cluster)')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.tight_layout()
    plt.savefig(output_path / 'types_tsne_clusters.png', dpi=300)
    print(f"  Saved: {output_path / 'types_tsne_clusters.png'}")
    
    # Plot 2: Density heatmap
    plt.figure(figsize=(12, 8))
    plt.hexbin(embeddings_2d[:, 0], embeddings_2d[:, 1], gridsize=30, cmap='YlOrRd')
    plt.colorbar(label='Density')
    plt.title('Anomaly Types - Density Distribution')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.tight_layout()
    plt.savefig(output_path / 'types_density.png', dpi=300)
    print(f"  Saved: {output_path / 'types_density.png'}")
    
    # Save embeddings and metadata
    np.save(output_path / 'embeddings.npy', embeddings)
    np.save(output_path / 'embeddings_2d.npy', embeddings_2d)
    np.save(output_path / 'labels.npy', labels)
    
    with open(output_path / 'types.json', 'w') as f:
        json.dump(types, f, indent=2)
    
    print(f"\n  All data saved to: {output_path}/")

def recommend_reward_scale(distances: np.ndarray):
    """Recommend reward scale based on distance analysis."""
    print("\n" + "="*80)
    print("REWARD SCALE RECOMMENDATIONS")
    print("="*80)
    
    mean_dist = distances.mean()
    std_dist = distances.std()
    p25 = np.percentile(distances, 25)
    p75 = np.percentile(distances, 75)
    
    print(f"\nBased on semantic distance analysis:")
    print(f"  Mean distance: {mean_dist:.4f}")
    print(f"  Std distance:  {std_dist:.4f}")
    print(f"  25th-75th:     {p25:.4f} - {p75:.4f}")
    
    print(f"\nRecommended reward scaling:")
    print(f"  1. STRICT (IAD-R1 style):")
    print(f"     - Exact match:      1.00")
    print(f"     - Very similar:     0.85  (distance < {p25:.3f})")
    print(f"     - Similar:          0.60  (distance < {mean_dist:.3f})")
    print(f"     - Somewhat similar: 0.40  (distance < {p75:.3f})")
    print(f"     - Different:        0.00")
    
    print(f"\n  2. LENIENT (Nomic semantic):")
    print(f"     - Cosine similarity as reward:")
    print(f"     - reward = max(0, 1 - distance)")
    print(f"     - This gives smooth gradient from 0.0 to 1.0")
    
    print(f"\n  3. HYBRID (Recommended):")
    print(f"     - Exact match:      1.00")
    print(f"     - Semantic (Nomic): 0.85 * (1 - distance)")
    print(f"     - Fuzzy match:      0.40 * string_similarity")
    print(f"     - This combines symbolic + semantic matching")
    
    # Specific thresholds
    high_sim_threshold = mean_dist - std_dist
    med_sim_threshold = mean_dist
    low_sim_threshold = mean_dist + std_dist
    
    print(f"\n  4. THRESHOLD-BASED:")
    print(f"     - Excellent: {1.00:.2f}  if distance < {high_sim_threshold:.3f}")
    print(f"     - Good:      {0.75:.2f}  if distance < {med_sim_threshold:.3f}")
    print(f"     - Fair:      {0.50:.2f}  if distance < {low_sim_threshold:.3f}")
    print(f"     - Poor:      {0.00:.2f}  otherwise")

# ==============================================================================
# Main
# ==============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze anomaly type distribution")
    parser.add_argument("--dataset", type=str, 
                       default=(os.environ.get('WORK_DIR','/bulk/aacudad/reasoning_traces')+"/Training/datasets/unified_train_oneshot_full.json"),
                       help="Path to unified dataset")
    parser.add_argument("--output-dir", type=str, default="anomaly_type_analysis",
                       help="Output directory for results")
    parser.add_argument("--n-clusters", type=int, default=10,
                       help="Number of clusters for K-means")
    parser.add_argument("--max-types", type=int, default=None,
                       help="Max number of types to analyze (for testing)")
    args = parser.parse_args()
    
    # Extract types
    types, metadata = extract_types_from_dataset(args.dataset)
    
    if not types:
        print("ERROR: No types found in dataset!")
        return
    
    # Limit for testing
    if args.max_types and len(types) > args.max_types:
        print(f"\nLimiting to {args.max_types} types for testing...")
        indices = np.random.choice(len(types), args.max_types, replace=False)
        types = [types[i] for i in indices]
        metadata = [metadata[i] for i in indices]
    
    # Analyze distribution
    type_counts = analyze_type_distribution(types, metadata)
    
    # Load Nomic model
    print("\n" + "="*80)
    print("LOADING NOMIC EMBED MODEL")
    print("="*80)
    nomic = NomicEmbedModel(device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Embed all types
    print("\n" + "="*80)
    print("EMBEDDING TYPES")
    print("="*80)
    embeddings = nomic.embed(types)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Compute distances
    distance_matrix, similarity_matrix = compute_distance_statistics(embeddings, types)
    
    # Perform clustering
    labels = perform_clustering(embeddings, types, n_clusters=args.n_clusters)
    
    # Visualize
    visualize_embeddings(embeddings, types, labels, output_dir=args.output_dir)
    
    # Recommend reward scale
    triu_indices = np.triu_indices_from(distance_matrix, k=1)
    distances = distance_matrix[triu_indices]
    recommend_reward_scale(distances)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {args.output_dir}/")
    print(f"  - types_tsne_clusters.png  : t-SNE visualization colored by cluster")
    print(f"  - types_density.png        : Density heatmap")
    print(f"  - embeddings.npy           : Full embeddings")
    print(f"  - types.json               : List of all types")

if __name__ == "__main__":
    main()
