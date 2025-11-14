import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.metrics import r2_score, mean_absolute_error

# ------------------ Test Evaluation and Visualization Functions ------------------

def evaluate_test_metrics(config):
    """
    Recompute metrics on the test set using the same logic as in train_modelwithGAT:
    - R² and MAE are computed on normalized continuous values (without inverse transformation)
    - OA is classification accuracy (argmax prediction vs. true label)
    - No floor/round/inverse_transform applied during metric calculation
    """
    model_path = config.get('save_path')
    output_dir = config.get('output_dir')
    save_csv = config.get('save_csv', False)
    dataloader = config.get('dataloader')
    device = config.get('device', 'cuda')
    means = config.get('mean')
    stds = config.get('std')

    # --------------------------- Load Model Weights ---------------------------
    print("🚀 Loading trained model weights...")
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        train_config = checkpoint['config']
    except Exception as e:
        raise FileNotFoundError(f"Failed to load model weights: {model_path}\nError: {e}")

    # --------------------------- Initialize Models ---------------------------
    fusion_type = train_config.get('fusion_type', 'concat')
    use_gat = train_config.get('use_GAT')
    gat_use_res = True

    from Multimodal_Semantic_Enhancer import Multimodal_Semantic_Enhancer
    from GATNetwork import MultiModalGNN

    model = Multimodal_Semantic_Enhancer(
        fusion_num=config.get('fusion_num'),
        feature_dim=256,
        fusion_type=fusion_type
    ).to(device)

    gat_model = MultiModalGNN(
        feature_dim=train_config['GAT_dim'],
        gat_hidden_dim=512,
        gat_heads=4,
        fusion_type=fusion_type,
        bin_centers=torch.tensor(config['bin_centers'], device=device),
        use_res=gat_use_res,
        classifier_arch=config.get('classifier_arch', 'transformer'),
        use_gnn='GAT'
    ).to(device)

    model.load_state_dict(checkpoint['model'])
    gat_model.load_state_dict(checkpoint['gat_model'])
    model.eval()
    gat_model.eval()
    print("🚀 Model weights loaded successfully...")

    # --------------------------- Load Graph Structure ---------------------------
    graph_data = torch.load()
    edge_index_full = graph_data['edge_index'].to(device)
    node_id_to_idx = graph_data['node_id_to_idx']

    # --------------------------- Retrieve Batch Data ---------------------------
    data_batch = next(iter(dataloader))
    gwbh_order = data_batch['gwbh'].cpu().tolist()
    num_nodes = len(gwbh_order)

    keep_node_indices = torch.tensor([
        node_id_to_idx[gwbh] for gwbh in gwbh_order
    ], dtype=torch.long, device=device)

    edge_index_sub, _ = subgraph(
        subset=keep_node_indices,
        edge_index=edge_index_full,
        relabel_nodes=True,
        num_nodes=len(node_id_to_idx)
    )

    data = Data(
        edge_index=edge_index_sub,
        y_class=data_batch['targets']['labels'].squeeze().to(device),   # [N, 24]
        y_reg=data_batch['targets']['normalized'].squeeze().to(device), # [N, 24] ← normalized values
        num_nodes=num_nodes
    ).to(device)

    # --------------------------- Split Test Set ---------------------------
    torch.manual_seed(train_config.get('seed', 42))
    np.random.seed(train_config.get('seed', 42))
    perm = torch.randperm(num_nodes)
    train_size = int(num_nodes * train_config.get('train_ratio', 0.6))
    val_size = int(num_nodes * train_config.get('val_ratio', 0.2))
    test_indices = perm[train_size + val_size:]
    data.test_mask = test_indices

    # --------------------------- Prepare Multimodal Inputs ---------------------------
    use_img = train_config.get('use_images', False)
    use_pois = train_config.get('use_pois', False)
    use_svis = train_config.get('use_svis', False)
    use_building = train_config.get('use_building', False)
    use_text = train_config.get('use_text', False)

    imgs = data_batch['img_code']['rs_code'].to(device) if use_img else None
    pois = data_batch['pois'].to(device) if use_pois else None
    svis = data_batch['svi'].to(device) if use_svis else None
    building_imgs = data_batch['img_code']['building_code'].to(device) if use_building else None
    text = {k: v.to(device) for k, v in data_batch['text'].items()} if use_text else None

    # --------------------------- Inference ---------------------------
    with torch.no_grad():
        modal_features = model(imgs, pois, building_imgs, svis, text, only_feature=True)

        if use_gat:
            logits, _, _ = gat_model(modal_features, data, return_contrastive_loss=False, return_embeddings=False, use_GNN=True)
        else:
            logits, _, _ = gat_model(modal_features, data, return_contrastive_loss=False, return_embeddings=False, use_GNN=False)

        # Extract test nodes
        test_mask = data.test_mask
        test_logits = logits[test_mask]              # [B, 24, C]
        test_y_reg_norm = data.y_reg[test_mask]      # [B, 24] ← normalized regression targets
        test_labels = data.y_class[test_mask]        # [B, 24] ← classification labels

        B, T, C = test_logits.shape
        bin_centers = torch.tensor(config['bin_centers'], device=device)

        # --- Regression prediction: weighted sum after softmax ---
        probs_flat = torch.softmax(test_logits.reshape(-1, C), dim=1)           # [B*T, C]
        reg_preds_flat = torch.sum(probs_flat * bin_centers, dim=1)             # [B*T]
        reg_preds = reg_preds_flat.view(B, T).cpu().numpy()                     # [B, 24]
        y_true_norm = test_y_reg_norm.cpu().numpy()                             # [B, 24]

        # --- Classification prediction ---
        pred_classes = torch.argmax(test_logits, dim=-1).cpu().numpy()          # [B, 24]
        true_classes = test_labels.cpu().numpy()                                # [B, 24]

    # --------------------------- Compute Hourly Metrics (Aligned with Training Code) ---------------------------
    print("📊 Computing per-hour metrics (consistent with training)...")

    hours = [f"H{i:02d}" for i in range(T)]
    results_r2 = []
    results_mae = []
    results_oa = []
    results_mape = []

    for t in range(T):
        # R² and MAE: computed on normalized continuous values (no inverse transform)
        r2 = r2_score(y_true_norm[:, t], reg_preds[:, t])
        mae = mean_absolute_error(y_true_norm[:, t], reg_preds[:, t])

        # MAPE (after inverse normalization + filtering zero values)
        y_true_nonstd = y_true_norm[:, t] * stds + means
        reg_preds_nonstd = reg_preds[:, t] * stds + means
        mask = y_true_nonstd > 0
        mape = np.mean(np.abs(reg_preds_nonstd[mask] - y_true_nonstd[mask]) / (y_true_nonstd[mask]))

        # OA: classification accuracy (class index match)
        oa = (true_classes[:, t] == pred_classes[:, t]).mean()

        results_r2.append(r2)
        results_mae.append(mae)
        results_oa.append(oa)
        results_mape.append(mape)

    # Construct DataFrames
    df_r2 = pd.DataFrame([results_r2], index=['R²'], columns=hours)
    df_mae = pd.DataFrame([results_mae], index=['MAE'], columns=hours)
    df_oa = pd.DataFrame([results_oa], index=['OA'], columns=hours)
    df_mape = pd.DataFrame([results_mape], index=['MAPE'], columns=hours)

    # --------------------------- Format and Output Results ---------------------------

    # Create display-ready copies
    df_r2_display = df_r2.copy()
    df_oa_display = df_oa.copy()
    df_mae_display = df_mae.copy()
    df_mape_display = df_mape.copy()

    # R²: round to 3 decimal places
    df_r2_display.loc['R²'] = df_r2.values[0].round(3)

    # OA: round to 3 decimal places
    df_oa_display.loc['OA'] = df_oa.values[0].round(3)

    # MAE: round to 3 decimal places
    df_mae_display.loc['MAE'] = df_mae.values[0].round(3)

    # MAPE: round to 3 decimal places
    df_mape_display.loc['MAPE'] = df_mape.values[0].round(3)

    # Print formatted results
    print("\n📈 R² per hour:")
    print(df_r2_display)

    print("\n🎯 OA per hour:")
    print(df_oa_display)

    print("\n📉 MAE per hour:")
    print(df_mae_display)

    print("\n📉 MAPE per hour:")
    print(df_mape_display)

    # Save to CSV — save formatted versions
    if save_csv:
        df_r2_display.to_csv(os.path.join(output_dir, "test_r2_per_hour.csv"), index_label="Metric")
        df_oa_display.to_csv(os.path.join(output_dir, "test_oa_per_hour.csv"), index_label="Metric")
        df_mae_display.to_csv(os.path.join(output_dir, "test_mae_per_hour.csv"), index_label="Metric")
        df_mape_display.to_csv(os.path.join(output_dir, "test_mape_per_hour.csv"), index_label="Metric")
        print(f"✅ All formatted results saved to: {output_dir}")

    return df_r2, df_oa, df_mae, df_mape


def plot_metrics_nature_style(r2_csv, mae_csv, oa_csv, mape_csv, output_dir=None, figsize=(10, 8)):
    """
    Load four metric CSV files and plot them in a 2x2 grid of line charts with enhanced visibility.
    Dynamically adjust Y-axis limits to avoid excessive whitespace.

    Args:
        r2_csv (str): Path to R² metrics CSV file
        mae_csv (str): Path to MAE metrics CSV file
        oa_csv (str): Path to OA metrics CSV file
        mape_csv (str): Path to MAPE metrics CSV file
        output_dir (str, optional): Directory to save the figure. If None, only display.
        figsize (tuple): Figure size (default: 10, 8)
    """
    # --------------------------- Load Data ---------------------------
    def load_metric(csv_path, metric_name):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"File not found: {csv_path}")
        df = pd.read_csv(csv_path, index_col=0)
        if metric_name not in df.index:
            raise ValueError(f"Metric '{metric_name}' not found in CSV: {csv_path}")
        return df.loc[metric_name].values  # Extract metric values

    try:
        r2_vals = load_metric(r2_csv, 'R²')
        mae_vals = load_metric(mae_csv, 'MAE')
        oa_vals = load_metric(oa_csv, 'OA')
        mape_vals = load_metric(mape_csv, 'MAPE')
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return

    # Ensure 24 time steps
    T = len(r2_vals)
    hours = np.arange(T)  # [0, 1, ..., 23]

    # --------------------------- Set Enhanced Style ---------------------------
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 28,              # Base font size increased
        "axes.titlesize": 28,
        "axes.labelsize": 28,         # Y-label font size increased
        "xtick.labelsize": 28,        # X-tick font size increased
        "ytick.labelsize": 28,        # Y-tick font size increased
        "legend.fontsize": 28,
        "axes.linewidth": 3,        # Axis spine width increased
        "xtick.major.width": 2,
        "ytick.major.width": 2,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "lines.linewidth": 12,       # Main line thickness increased
        "lines.markersize": 20,
        "figure.figsize": figsize,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # Define colors
    colors = {
        'R²': '#1f77b4',      # blue
        'MAE': '#d62728',     # red
        'OA': '#2ca02c',      # green
        'MAPE': '#ff7f0e',    # orange
    }

    # --------------------------- Create Subplots ---------------------------
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # --- Helper function for dynamic Y-limits ---
    def set_y_limits(ax, data, min_val=None, max_val=None, padding=0.1):
        data_min = np.min(data)
        data_max = np.max(data)
        data_range = data_max - data_min
        pad = data_range * padding

        y_min = data_min - pad if min_val is None else max(min_val, data_min - pad)
        y_max = data_max + pad if max_val is None else min(max_val, data_max + pad)

        if y_min >= y_max:
            y_min = data_min - 0.05 * data_range
            y_max = data_max + 0.05 * data_range

        ax.set_ylim(y_min, y_max)

    # --- X-tick settings: show every 3rd hour ---
    x_ticks = np.arange(0, 24, 3)  # [0, 3, 6, ..., 21]

    # --- R² ---
    axes[0, 0].plot(hours, r2_vals, color=colors['R²'], marker='o', 
                    markerfacecolor=colors['R²'], markeredgecolor=colors['R²'], 
                    markeredgewidth=1.5, zorder=3)
    axes[0, 0].fill_between(hours, r2_vals, color=colors['R²'], alpha=0.25, zorder=2)
    axes[0, 0].set_ylabel("R²", fontsize=60, labelpad=25)
    set_y_limits(axes[0, 0], r2_vals, min_val=0.6, max_val=1.0)
    axes[0, 0].grid(True, axis='both', linestyle='--', alpha=0.4, linewidth=3, zorder=0)

    # --- MAE ---
    axes[0, 1].plot(hours, mae_vals, color=colors['MAE'], marker='s',
                    markerfacecolor=colors['MAE'], markeredgecolor=colors['MAE'],
                    markeredgewidth=1.5, zorder=3)
    axes[0, 1].fill_between(hours, mae_vals, color=colors['MAE'], alpha=0.25, zorder=2)
    axes[0, 1].set_ylabel("MAE", fontsize=60, labelpad=25)
    set_y_limits(axes[0, 1], mae_vals, min_val=0, max_val=None)
    axes[0, 1].grid(True, axis='both', linestyle='--', alpha=0.4, linewidth=3, zorder=0)

    # --- OA ---
    axes[1, 0].plot(hours, oa_vals, color=colors['OA'], marker='^',
                    markerfacecolor=colors['OA'], markeredgecolor=colors['OA'],
                    markeredgewidth=1.5, zorder=3)
    axes[1, 0].fill_between(hours, oa_vals, color=colors['OA'], alpha=0.25, zorder=2)
    axes[1, 0].set_ylabel("OA", fontsize=60, labelpad=25)
    set_y_limits(axes[1, 0], oa_vals, min_val=0.45, max_val=1.0)
    axes[1, 0].grid(True, axis='both', linestyle='--', alpha=0.4, linewidth=3, zorder=0)

    # --- MAPE ---
    axes[1, 1].plot(hours, mape_vals, color=colors['MAPE'], marker='D',
                    markerfacecolor=colors['MAPE'], markeredgecolor=colors['MAPE'],
                    markeredgewidth=1.5, zorder=3)
    axes[1, 1].fill_between(hours, mape_vals, color=colors['MAPE'], alpha=0.25, zorder=2)
    axes[1, 1].set_ylabel("MAPE", fontsize=60, labelpad=25)
    set_y_limits(axes[1, 1], mape_vals, min_val=0, max_val=None)
    axes[1, 1].grid(True, axis='both', linestyle='--', alpha=0.4, linewidth=3, zorder=0)

    # Common x-axis settings: only show every 3rd tick
    for ax in axes.flat:
        ax.set_xlabel("Hour of Day", labelpad=25, fontsize=45)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f"{h}" for h in x_ticks], rotation=0)

    # Adjust layout
    plt.tight_layout()

    # --------------------------- Save Plot ---------------------------
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_path_png = os.path.join(output_dir, "error_time.png")
        save_path_pdf = os.path.join(output_dir, "error_time.pdf")
        plt.savefig(save_path_png)
        plt.savefig(save_path_pdf)
        print(f"✅ Figures saved to:\n   {save_path_png}\n   {save_path_pdf}")

    plt.show()