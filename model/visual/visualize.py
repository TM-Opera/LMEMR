import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn.manifold import TSNE
import os

# ------------------ Core Plotting Functions ------------------

def plot_confusion_matrix_both(cm, classes=range(0, 10),
                               title='Confusion Matrix',
                               cmap=plt.cm.Blues,
                               save_path=None):
    """
    Plot both normalized and unnormalized confusion matrices side by side (dual subplots).

    Args:
        cm: Confusion matrix as a numpy.ndarray
        classes: List of class labels
        title: Title of the plot
        cmap: Colormap scheme
        save_path: If specified, saves the figure to this path
    """
    # Create figure and subplots
    fig, axes = plt.subplots(1, 2, figsize=(18, 9), dpi=100)

    # Unnormalized confusion matrix
    im1 = sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                      xticklabels=classes, yticklabels=classes,
                      annot_kws={"size": 10}, ax=axes[0], square=True)
    axes[0].set_title(f'{title} (Unnormalized)', fontsize=14, pad=20)
    axes[0].set_xlabel('Predicted Label', fontsize=12)
    axes[0].set_ylabel('True Label', fontsize=12)
    axes[0].tick_params(axis='both', which='major', labelsize=10)
    axes[0].set_xticklabels(classes, rotation=45, ha='right')
    axes[0].set_yticklabels(classes, rotation=0, va='center')

    # Normalized confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    im2 = sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap=cmap,
                      xticklabels=classes, yticklabels=classes,
                      annot_kws={"size": 10}, ax=axes[1], square=True,
                      cbar_kws={'label': 'Normalized Value'})
    axes[1].set_title(f'{title} (Normalized)', fontsize=14, pad=20)
    axes[1].set_xlabel('Predicted Label', fontsize=12)
    axes[1].set_ylabel('True Label', fontsize=12)
    axes[1].tick_params(axis='both', which='major', labelsize=10)
    axes[1].set_xticklabels(classes, rotation=45, ha='right')
    axes[1].set_yticklabels(classes, rotation=0, va='center')

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save image
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Image saved to: {save_path}")

    plt.show()


# t-SNE visualization of multi-modal node embeddings
def plot_tsne_embeddings(modalities_dict, 
                         labels=None, 
                         modality_names=None,
                         save_path=None, 
                         title="t-SNE of Multi-Modal Node Embeddings",
                         figsize=(12, 8),
                         perp=30.0):
    """
    Plot t-SNE visualization of multi-modal node embeddings.

    Args:
        modalities_dict: dict {mod_name: tensor[N_mod, d]}
        labels: optional, tensor[N_total] with class labels (aligned with concatenated nodes)
        modality_names: list of str, specify display order and aliases, e.g., ['rs', 'poi'] or ['Remote Sensing', 'POI']
                        If None, use modalities_dict.keys() with default color mapping
        save_path: path to save image (e.g., 'plots/tsne_epoch_10.png')
        title: plot title
        figsize: figure size
        perp: perplexity for t-SNE
    """
    # Collect all modality embeddings and mark their sources
    embeddings = []
    mod_labels = []      # Original modality key (for indexing)
    mod_display = []     # Display name (for legend)
    node_labels = []

    # Default color palette (extended for more modalities)
    default_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
        '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'
    ]
    
    # Determine modalities to plot and their display names
    if modality_names is not None:
        # If provided, may be alias list like ['RS', 'POI']
        display_map = {}
        keys_in_order = []
        for item in modality_names:
            found = False
            for k in modalities_dict.keys():
                if item == k or item in ['rs', 'poi', 'building', 'svi']:  # Rough match
                    display_map[k] = item
                    keys_in_order.append(k)
                    found = True
                    break
            if not found:
                # Skip or warn if no matching modality found
                print(f"Notice: Modality name '{item}' not found in modalities_dict keys.")
                pass
        final_mods = keys_in_order
        mod_to_display = display_map
    else:
        final_mods = list(modalities_dict.keys())
        mod_to_display = {k: k for k in final_mods}

    # Assign colors (cycling through)
    color_map = {}
    for idx, mod in enumerate(final_mods):
        color_map[mod] = default_colors[idx % len(default_colors)]

    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'P', '*']
    
    idx_offset = 0
    for mod_key in final_mods:
        if mod_key not in modalities_dict:
            continue
        z = modalities_dict[mod_key]
        if not isinstance(z, torch.Tensor):
            continue
        z = z.detach().cpu().numpy()
        n_nodes = z.shape[0]

        embeddings.append(z)
        mod_labels.extend([mod_key] * n_nodes)
        mod_display.extend([mod_to_display[mod_key]] * n_nodes)

        if labels is not None:
            lbls = labels[idx_offset:idx_offset + n_nodes].cpu().numpy()
            node_labels.extend(lbls)
        else:
            node_labels.extend([0] * n_nodes)
        idx_offset += n_nodes

    if len(embeddings) == 0:
        print("No valid embeddings to visualize.")
        return

    # Concatenate embeddings
    embeddings = np.vstack(embeddings)
    mod_display = np.array(mod_display)
    node_labels = np.array(node_labels)
    unique_mods = np.unique(mod_display)
    unique_classes = np.unique(node_labels) if labels is not None and len(node_labels) > 0 else []

    # t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, perplexity=perp, learning_rate='auto', random_state=42, init='pca', n_jobs=-1)
    embed_2d = tsne.fit_transform(embeddings)

    # Plot
    plt.figure(figsize=figsize)

    if labels is not None and len(unique_classes) > 1:
        # Use marker to distinguish classes, color to distinguish modalities
        for i, cls in enumerate(unique_classes):
            mask_cls = node_labels == cls
            for mod in unique_mods:
                mask_mod = mod_display == mod
                mask = mask_cls & mask_mod
                if not mask.any():
                    continue
                plt.scatter(embed_2d[mask, 0], embed_2d[mask, 1],
                            c=[color_map[next(k for k, v in mod_to_display.items() if v == mod)]],
                            marker=markers[i % len(markers)],
                            label=f'{mod} - Class {int(cls)}',
                            alpha=0.7, s=50, edgecolors='black')
    else:
        for mod in unique_mods:
            mask = mod_display == mod
            color = color_map[next(k for k, v in mod_to_display.items() if v == mod)]
            plt.scatter(embed_2d[mask, 0], embed_2d[mask, 1],
                        c=[color], label=mod,
                        alpha=0.7, s=50, edgecolors='black')

    # --- Style improvements ---
    # 1. Add grid
    plt.grid(True, linestyle='-', linewidth=0.5, alpha=0.7)
    # 2. Increase font size for axes (without bold)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # -----------------

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(title)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        #print(f"t-SNE plot saved to: {save_path}")
    else:
        plt.show()
    plt.close()



import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


def easy_visual_metrics_clean_fixed(metrics_paths, model_names, save_path=None, figsize=(18, 14)):
    """
    Final optimized version addressing:
    - First point of Training Loss being clipped
    - No padding on X-axis
    - Confidence interval too small to see
    - Lack of uncertainty representation
    - Added support for visualizing val_oa and val_mae
    - ✅ Improved: Increased subplot spacing + larger axis/title fonts
    """
    if len(metrics_paths) != len(model_names):
        raise ValueError("metrics_paths and model_names must have the same length")

    # Metrics to visualize
    metric_pairs = [
        ('train_loss', 'Training Loss'),
        ('val_r2', 'Validation R²'),
        ('val_oa', 'Validation OA'),
        ('val_mae', 'Validation MAE')
    ]

    y_labels = {
        'train_loss': 'Training Loss',
        'val_r2': 'R² Score',
        'val_oa': 'OA',
        'val_mae': 'MAE'
    }

    # Color and line style (academic style)
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b']
    linestyles = ['-', '--', '-.', ':', '-']

    # Create 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    all_train_losses = []

    for i, (metric_key, metric_label) in enumerate(metric_pairs):
        ax = axes[i]

        for j, (path, name) in enumerate(zip(metrics_paths, model_names)):
            try:
                metrics = np.load(path)
            except Exception as e:
                print(f"⚠️ Warning: Failed to load {path}, skipping this model. Error: {e}")
                continue

            if metric_key not in metrics:
                print(f"⚠️ Warning: Metric {metric_key} not found in {path}")
                continue

            data = metrics[metric_key].copy()
            epochs = np.arange(len(data))

            if metric_key == 'train_loss':
                all_train_losses.append(data)

            # === Smoothing ===
            window_size = 21
            polyorder = 3

            if len(data) >= window_size:
                smoothed = savgol_filter(data, window_length=window_size, polyorder=polyorder)
            else:
                w = min(window_size, len(data))
                if w % 2 == 0:
                    w -= 1
                smoothed = np.convolve(data, np.ones(w)/w, mode='valid')
                pad_len = (len(data) - len(smoothed)) // 2
                epochs = np.arange(pad_len, pad_len + len(smoothed))

            # ================================
            # 🔥 Improved confidence interval
            # ================================
            std_est = np.std(data) * 0.25
            upper_bound = smoothed + std_est
            lower_bound = smoothed - std_est

            color = colors[j % len(colors)]
            linestyle = linestyles[j % len(linestyles)]

            # Plot confidence band
            ax.fill_between(epochs, lower_bound, upper_bound,
                            color=color, alpha=0.25, edgecolor='none', label='_nolegend_')

            # Plot main curve
            ax.plot(epochs, smoothed,
                    label=name,
                    color=color,
                    linestyle=linestyle,
                    linewidth=6,
                    marker='o' if j % 2 == 0 else None,
                    markersize=10,
                    markevery=max(1, len(epochs) // 20))

        # Set subplot style —— ✅ Larger font sizes
        ax.set_xlabel('Epoch', fontsize=26, labelpad=10)
        ax.set_ylabel(y_labels[metric_key], fontsize=26, labelpad=10)
        ax.grid(True, linestyle='--', linewidth=2)
        ax.tick_params(axis='both', which='major', labelsize=23)

        # ✅ Padding on X-axis
        total_epochs = 300
        ax.set_xlim(-total_epochs * 0.02, total_epochs * 1.02)

        # ✅ Y-axis range settings
        if metric_key == 'train_loss':
            if len(all_train_losses) > 0:
                global_min = min(np.min(d) for d in all_train_losses)
                global_max = max(np.max(d) for d in all_train_losses)
                padding = (global_max - global_min) * 0.05
                ax.set_ylim(bottom=global_min - padding, top=global_max + padding)
            else:
                ax.set_ylim(bottom=0, top=3.0)
        elif metric_key == 'val_r2':
            ax.set_ylim(-0.5, 1.0)
        elif metric_key == 'val_oa':
            ax.set_ylim(0.0, 0.7)
        elif metric_key == 'val_mae':
            ax.set_ylim(bottom=0)

        # ✅ Border styling applied later for consistency

    # ================================
    # 🎨 Legend setup (centered at bottom, single row)
    # ================================
    handles = []
    for j, name in enumerate(model_names):
        handles.append(plt.Line2D([0], [0],
                                  label=name,
                                  color=colors[j % len(colors)],
                                  linestyle=linestyles[j % len(linestyles)],
                                  linewidth=3))

    fig.legend(handles=handles,
               loc='lower center',
               ncol=len(model_names),
               fontsize=26,
               frameon=False,
               edgecolor='gray',
               columnspacing=1.8,
               handlelength=2.5,
               bbox_to_anchor=(0.5, 0.02))

    # ================================
    # ✅ ✅ 【Critical Fix】Apply border styling to all subplots
    # ================================
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('black')
        ax.spines['bottom'].set_linewidth(2.5)
        ax.spines['left'].set_color('black')
        ax.spines['left'].set_linewidth(2.5)

    # Increase subplot spacing
    plt.subplots_adjust(hspace=0.8, wspace=0.35)

    # Adjust overall layout to make room for legend
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])

    # Save or show
    if save_path:
        if save_path.lower().endswith('.pdf'):
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', format='pdf')
            print(f"✅ PDF saved to: {save_path}")
        else:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', format='png')
            print(f"✅ Visualization saved to: {save_path}")
    else:
        plt.show()

    return fig, axes