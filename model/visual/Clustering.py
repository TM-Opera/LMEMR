import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import geopandas as gpd


# ------------------ Clustering Functions ------------------

# -------------------------------
# Configuration Parameters
# -------------------------------
main_folder = ''   
geojson_path = ''    
output_dir = 'cluster_results/cluster'
image_dir = 'cluster_results/image'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)

output_plot_elbow = os.path.join(image_dir, 'elbow_silhouette.png')
output_plot_clusters = os.path.join(image_dir, 'cluster_patterns_original_scale.png')
output_plot_k_comparison = os.path.join(image_dir, 'k_comparison_log_scaled.png')

spatial_plot_output = os.path.join(image_dir, 'spatial_cluster_comparison_by_k.png')
dendrogram_output = os.path.join(image_dir, 'dendrogram_agglomerative.png')  # Output path for dendrogram
linkage_matrix_path = os.path.join(output_dir, 'linkage_matrix.npy')         # Path to save linkage matrix

# -------------------------------
# Clustering Method Selection (toggleable)
# -------------------------------
CLUSTERING_METHOD = 'agglomerative'  # Supported: 'kmeans', 'agglomerative', 'gmm', 'dbscan'

clustering_algorithms = {
    'kmeans': lambda k: KMeans(n_clusters=k, random_state=42, n_init=10),
    'agglomerative': lambda k: AgglomerativeClustering(n_clusters=k),
    'gmm': lambda k: GaussianMixture(n_components=k, random_state=42, covariance_type='full'),
    'dbscan': lambda _: DBSCAN(eps=0.5, min_samples=5)
}

# -------------------------------
# Data Transformation Method Selection
# -------------------------------
def apply_transformation(X, method='l2_normalize'):
    """
    Apply a transformation to the data.
    
    Options:
        - 'log': log(1 + x)
        - 'row_zscore': Z-score normalization per row (per node)
        - 'log_row_zscore': log then row-wise z-score
        - 'l2_normalize': L2 normalization across features
    """
    if method == 'log':
        return np.log1p(X)
    elif method == 'row_zscore':
        X_normalized = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)
        return X_normalized
    elif method == 'log_row_zscore':
        X_log = np.log1p(X)
        X_normalized = (X_log - X_log.mean(axis=1, keepdims=True)) / (X_log.std(axis=1, keepdims=True) + 1e-8)
        return X_normalized
    elif method == 'l2_normalize':
        norm = np.linalg.norm(X, axis=1, keepdims=True)
        return X / (norm + 1e-8)
    else:
        raise ValueError("method must be 'log', 'row_zscore', 'log_row_zscore', or 'l2_normalize'")

TRANSFORMATION_METHOD = 'log'


# -------------------------------
# 1. Load All Traffic Flow Data
# -------------------------------
data_list = []
gwbh_list = []

print("Traversing folder and loading data...")
for item in os.listdir(main_folder):
    item_path = os.path.join(main_folder, item)
    if os.path.isdir(item_path) and item.isdigit():
        gwbh = item
        true_file = os.path.join(item_path, f'{gwbh}_flow.npy')
        if os.path.exists(true_file):
            try:
                flow_data = np.load(true_file)
                if flow_data.shape == (24,):
                    pass
                elif flow_data.shape == (1, 24):
                    flow_data = flow_data.flatten()
                else:
                    print(f"[Warning] Invalid shape for {gwbh}: {flow_data.shape}, skipped.")
                    continue
                data_list.append(flow_data)
                gwbh_list.append(gwbh)
            except Exception as e:
                print(f"[Error] Failed to load {true_file}: {e}")
        else:
            print(f"[Missing] No flow.npy found in {item_path}")

if len(data_list) == 0:
    raise ValueError("No valid data loaded!")

X = np.array(data_list)
print(f"Successfully loaded data from {X.shape[0]} regions, shape: {X.shape}")

if (X < 0).any():
    raise ValueError("Traffic flow data contains negative values, cannot apply log(1+x) transformation!")


# -------------------------------
# 2. Apply Selected Data Transformation
# -------------------------------
X_transformed = apply_transformation(X, method=TRANSFORMATION_METHOD)
print(f"Using transformation method: {TRANSFORMATION_METHOD}")
print(f"Original data range: [{X.min():.2f}, {X.max():.2f}]")
print(f"Transformed data range: [{X_transformed.min():.2f}, {X_transformed.max():.2f}]")


# -------------------------------
# 3. Elbow Plot & Silhouette Analysis (KMeans only)
# -------------------------------
if CLUSTERING_METHOD == 'kmeans':
    print("Performing elbow analysis (for reference only)...")
    k_range = range(1, 11)
    inertias = []
    silhouette_scores = []

    for k in k_range:
        if k == 1:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_transformed)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(float('nan'))
        else:
            model = clustering_algorithms['kmeans'](k)
            labels = model.fit_predict(X_transformed)
            inertias.append(model.inertia_)
            sil_score = silhouette_score(X_transformed, labels)
            silhouette_scores.append(sil_score)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    color = 'tab:blue'
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia', color=color)
    ax1.plot(k_range, inertias, 'bo-', label='Inertia', markersize=6)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Silhouette Score', color=color)
    ax2.plot(k_range, silhouette_scores, 'r*--', label='Silhouette', markersize=6)
    ax2.tick_params(axis='y', labelcolor=color)
    plt.title('Elbow Method and Silhouette Analysis\n(for reference only)')
    fig.tight_layout()
    plt.savefig(output_plot_elbow, dpi=150)
    plt.close()


# -------------------------------
# 4. Perform Clustering (k = 2 to 9)
# -------------------------------
precomputed_labels = {}
result_dfs = {}

if CLUSTERING_METHOD == 'dbscan':
    print(f"Running {CLUSTERING_METHOD.upper()} clustering...")
    labels = clustering_algorithms['dbscan'](None).fit_predict(X_transformed)
    unique_clusters = np.unique(labels)
    print(f"DBSCAN found {len(unique_clusters)} clusters (including noise): {unique_clusters}")

    result_df = pd.DataFrame({'gwbh': gwbh_list, 'Cluster': labels})
    result_df.to_csv(os.path.join(output_dir, 'cluster_results_dbscan.csv'), index=False)
    print(f"✅ DBSCAN results saved to: cluster_results_dbscan.csv")

else:
    print("Clustering for k = 2 to 9...")
    for k in range(2, 10):
        model = clustering_algorithms[CLUSTERING_METHOD](k)
        if CLUSTERING_METHOD == 'gmm':
            labels = model.fit_predict(X_transformed)
        else:
            labels = model.fit_predict(X_transformed)
        precomputed_labels[k] = labels
        result_df = pd.DataFrame({'gwbh': gwbh_list, 'Cluster': labels})
        csv_path = os.path.join(output_dir, f'cluster_results_{CLUSTERING_METHOD}_k{k}.csv')
        result_df.to_csv(csv_path, index=False)
        result_dfs[k] = result_df
        print(f"✅ Clustering completed for k={k}, results saved to: {csv_path}")


# -------------------------------
# Generate Dendrogram with Bold Lines and Large Fonts
# -------------------------------
if CLUSTERING_METHOD == 'agglomerative':
    from scipy.cluster.hierarchy import dendrogram, linkage
    import matplotlib.pyplot as plt

    print("Generating hierarchical clustering dendrogram and saving linkage matrix...")

    # Parameters
    LINKAGE_METHOD = 'average'
    METRIC = 'euclidean'

    # Compute linkage matrix
    if LINKAGE_METHOD == 'ward':
        Z = linkage(X_transformed, method=LINKAGE_METHOD, metric=METRIC)
    else:
        from scipy.spatial.distance import pdist
        distances = pdist(X_transformed, metric=METRIC)
        Z = linkage(distances, method=LINKAGE_METHOD)

    # Save linkage matrix
    np.save(linkage_matrix_path, Z)
    print(f"🔗 Linkage matrix saved to: {linkage_matrix_path}")

    # Plot dendrogram with bold lines and large fonts
    plt.figure(figsize=(14, 12))

    with plt.rc_context({'lines.linewidth': 6}):  # Thicker lines
        dendrogram(
            Z,
            truncate_mode='lastp',
            p=20,
            show_leaf_counts=True,
            leaf_rotation=90.,
            leaf_font_size=5,
            show_contracted=True,
        )

    ax = plt.gca()

    # ✅ Make axis spines thicker
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(3)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Increase font sizes
    plt.xlabel('Cluster Size', fontsize=24, fontweight='bold')
    plt.ylabel('Distance', fontsize=24, fontweight='bold')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.tight_layout()
    plt.savefig(dendrogram_output, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"🌳 Hierarchical clustering dendrogram saved to: {dendrogram_output}")


# -------------------------------
# 5. Visualize Average Traffic Patterns (Original Scale)
# -------------------------------
if CLUSTERING_METHOD != 'dbscan':
    plt.figure(figsize=(12, 7))
    hours = np.arange(24)
    example_k = 9
    result_df = result_dfs[example_k]
    X_original = X  # Use original scale

    for cluster_id in sorted(result_df['Cluster'].unique()):
        mask = result_df['Cluster'] == cluster_id
        original_in_cluster = X_original[mask]
        mean_series = original_in_cluster.mean(axis=0)
        std_series = original_in_cluster.std(axis=0)
        count = len(original_in_cluster)
        plt.plot(hours, mean_series, linewidth=2.5, label=f'Cluster {cluster_id} (n={count})')
        plt.fill_between(hours, mean_series - std_series, mean_series + std_series, alpha=0.2)

    plt.xticks(hours)
    plt.xlabel('Hour of Day')
    plt.ylabel('Traffic Flow (Original Scale)')
    plt.title(f'Average Hourly Traffic Patterns by Cluster (k={example_k})')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_plot_clusters, dpi=150)
    plt.close()
    print(f"📈 Time-series pattern plot saved to: {output_plot_clusters}")


# -------------------------------
# 6. PCA Comparison Plot (Different k Values)
# -------------------------------
def plot_k_comparison_with_precomputed(X_input, precomputed_labels, output_path):
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_input)
    explained_ratio = pca.explained_variance_ratio_.sum()
    print(f"PCA explained variance ratio: {explained_ratio:.2%}")

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.ravel()

    # First subplot: raw data
    ax = axes[0]
    ax.scatter(X_2d[:, 0], X_2d[:, 1], c='blue', s=20, edgecolors='black', linewidth=0.2, alpha=0.7)
    ax.set_title('Dataset (PCA 2D Projection)')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.grid(True, alpha=0.3)

    # Subsequent subplots: clustering results
    for i, k in enumerate(range(2, 10)):
        ax = axes[i + 1]
        labels = precomputed_labels[k]
        colors = cm.get_cmap('tab10')(np.linspace(0, 1, k))[:k]
        for j in range(k):
            mask = labels == j
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[colors[j]], s=20, label=f'C{j}', alpha=0.7, edgecolors='black', linewidth=0.2)
        ax.set_title(f'K = {k}')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Clustering Results under Different k Values (PCA Visualization)', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"📊 PCA comparison plot saved to: {output_plot_k_comparison}")

if CLUSTERING_METHOD != 'dbscan':
    plot_k_comparison_with_precomputed(X_transformed, precomputed_labels, output_plot_k_comparison)


# -------------------------------
# 7. Write Results to GeoJSON
# -------------------------------
print("Loading GeoJSON file...")
try:
    gdf = gpd.read_file(geojson_path)
except Exception as e:
    raise ValueError(f"Failed to read GeoJSON file {geojson_path}: {e}")

if 'GWBH_500' not in gdf.columns:
    raise ValueError(f"GeoJSON file missing 'GWBH_500' column. Available columns: {list(gdf.columns)}")

gdf['GWBH_500'] = gdf['GWBH_500'].astype(str).str.strip()

if CLUSTERING_METHOD != 'dbscan':
    for k in range(2, 10):
        temp_result = result_dfs[k][['gwbh', 'Cluster']].copy()
        temp_result['gwbh'] = temp_result['gwbh'].astype(str).str.strip()
        temp_result.rename(columns={'Cluster': f'Cluster_k{k}'}, inplace=True)
        gdf = gdf.merge(temp_result, left_on='GWBH_500', right_on='gwbh', how='left').drop(columns=['gwbh'], errors='ignore')

    output_geojson = os.path.join(output_dir, f'clustered_regions_{CLUSTERING_METHOD}_all_k.geojson')
    gdf.to_file(output_geojson, driver='GeoJSON')
    print(f"✅ All k-value clustering results merged and saved to GeoJSON: '{output_geojson}'")


# -------------------------------
# 8. Plot Spatial Distribution of Clusters
# -------------------------------
def plot_spatial_clusters_by_k(gdf, precomputed_labels, gwbh_list, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    axes = axes.ravel()

    cmap = ListedColormap([cm.get_cmap('tab10')(i) for i in range(10)])
    gwbh_series = pd.Series(gwbh_list).astype(str).str.strip()

    # First subplot: base map
    ax = axes[0]
    gdf.plot(ax=ax, color='lightgray', edgecolor='black', linewidth=0.2)
    ax.set_title('Original Map', fontsize=14, pad=10)
    ax.axis('off')

    # Remaining subplots: clustering results
    for i, k in enumerate(range(2, 10)):
        ax = axes[i + 1]
        labels = precomputed_labels[k]
        temp_result = pd.DataFrame({'gwbh': gwbh_series.values, 'Cluster': labels})
        temp_result['Cluster'] = temp_result['Cluster'].astype(int)
        gdf_with_cluster = gdf.merge(temp_result, left_on='GWBH_500', right_on='gwbh', how='left')

        gdf_with_cluster.plot(
            column='Cluster',
            ax=ax,
            cmap=cmap,
            legend=False,
            edgecolor='black',
            linewidth=0.2
        )
        ax.set_title(f'K = {k}', fontsize=14, pad=10)
        ax.axis('off')

    plt.suptitle('Spatial Clustering Results (k = 2 to 9)', fontsize=18, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"🌍 Spatial distribution plot saved to: {spatial_plot_output}")

if CLUSTERING_METHOD != 'dbscan':
    plot_spatial_clusters_by_k(gdf, precomputed_labels, gwbh_list, spatial_plot_output)

print("✅ All tasks completed!")