import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import r2_score, mean_absolute_error, confusion_matrix, precision_score, recall_score, f1_score
import numpy as np
from torch_geometric.data import Data, DataLoader as GeoDataLoader
from torch_geometric.utils import subgraph
import time
import os
from GATNetwork import MultiModalGNN
from loss_module import ContrastiveLoss, ClassificationRegressionLoss
from visual.visualize import plot_tsne_embeddings
from torch.amp import GradScaler

# ------------------ Training Function and Metrics Computation ------------------

def train_modelwithGAT(model, full_loader, config):
    """
    Train the GNN model: end-to-end joint training with support for contrastive learning.
    All data is loaded at once (full-batch), where batch_size == total number of nodes in the graph.
    
    Args:
        model: The feature extraction model
        full_loader: DataLoader containing multimodal input data
        config: Configuration dictionary with training parameters
    
    Returns:
        Trained feature extraction model
    """
    device = config['device']
    torch.manual_seed(config.get('seed', 42))
    np.random.seed(config.get('seed', 42))

    # --------------------------- Load Graph Structure ---------------------------
    graph_data = torch.load(config['edge_data'])
    edge_index_full = graph_data['edge_index'].to(device)        # [2, E]
    num_nodes_full = graph_data['num_nodes']
    node_id_to_idx = graph_data['node_id_to_idx']               # gwbh -> graph index

    # --------------------------- Retrieve Single Batch ---------------------------
    print("🚀 Loading data...")
    data_batch = next(iter(full_loader))
    bin_centers = torch.tensor(config['bin_centers'], device=device, dtype=torch.float32).detach()

    # Extract node ID order (for indexing into the graph)
    gwbh_order = data_batch['gwbh'].cpu().tolist()
    num_nodes = len(gwbh_order)

    print(f"📌 Number of data nodes: {num_nodes} | Original graph nodes: {num_nodes_full}")

    if len(set(gwbh_order)) != len(gwbh_order):
        raise ValueError("Duplicate node IDs found in gwbh_order!")

    # --------------------------- Build Subgraph Node Indices ---------------------------
    keep_node_indices = []
    keep_gwbh_list = []
    for gwbh in gwbh_order:
        if gwbh not in node_id_to_idx:
            raise KeyError(f"Node {gwbh} not found in graph's node_id_to_idx!")
        keep_node_indices.append(node_id_to_idx[gwbh])
        keep_gwbh_list.append(gwbh)
    keep_node_indices = torch.tensor(keep_node_indices, dtype=torch.long, device=device)

    print(f"✅ Keeping {num_nodes} nodes, reconstructing subgraph...")
    edge_index_sub, _ = subgraph(
        subset=keep_node_indices,
        edge_index=edge_index_full,
        relabel_nodes=True,
        num_nodes=num_nodes_full
    )

    # Create new mapping: subgraph index corresponds directly to the order in gwbh list
    subgraph_idx_to_node_id = {i: gwbh for i, gwbh in enumerate(keep_gwbh_list)}

    print(f"✅ Subgraph constructed: Nodes {num_nodes} | Edges {edge_index_sub.size(1)}")

    # --------------------------- Construct PyG Data Object ---------------------------
    processor = config['processor']   # Data processor / for inverse normalization
    data = Data(
        edge_index=edge_index_sub,
        y_class=data_batch['targets']['labels'].squeeze(),        # [N, 24]
        y_reg=data_batch['targets']['normalized'].squeeze(),      # [N, 24]
        num_nodes=num_nodes
    ).to(device)

    # --------------------------- Create MASKs (Train/Validation/Test) ---------------------------
    train_ratio = config.get('train_ratio', 0.6)
    val_ratio = config.get('val_ratio', 0.2)
    test_ratio = 1.0 - train_ratio - val_ratio
    assert test_ratio > 0, "Train + validation ratio exceeds 1.0"

    perm = torch.randperm(num_nodes)
    train_size = int(num_nodes * train_ratio)
    val_size = int(num_nodes * val_ratio)

    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)

    data.train_mask[perm[:train_size]] = True
    data.val_mask[perm[train_size:train_size + val_size]] = True
    data.test_mask[perm[train_size + val_size:]] = True

    print(f"✅ Data split: Train {data.train_mask.sum():3d} | Val {data.val_mask.sum():3d} | Test {data.test_mask.sum():3d}")

    # --------------------------- Build GAT Network ---------------------------
    fusion_type = config.get('fusion_type', 'concat')
    num_modalities = config.get('num_modalities', 4)
    
    if config['use_GAT'] == 'resGAT':
        print("✅ Using ResGAT...")
        gat_model = MultiModalGNN(
            feature_dim=config['GAT_dim'],
            gat_hidden_dim=512,
            gat_heads=4,
            num_modalities=num_modalities,
            fusion_type=fusion_type,
            bin_centers=bin_centers,
            use_res=True,
            classifier_arch=config.get('classifier_arch', None),
            use_gnn='GAT'
        ).to(device)
        config['use_GAT'] = True
        config['GNN_type'] = 'resGAT'

    elif config['use_GAT'] == 'GAT':
        print("✅ Using GAT...")
        gat_model = MultiModalGNN(
            feature_dim=config['GAT_dim'],
            gat_hidden_dim=512,
            gat_heads=4,
            num_modalities=num_modalities,
            fusion_type=fusion_type,
            bin_centers=bin_centers,
            use_res=False,
            classifier_arch=config.get('classifier_arch', None),
            use_gnn='GAT'
        ).to(device)
        config['use_GAT'] = True
        config['GNN_type'] = 'GAT'

    elif config['use_GAT'] == 'GCN':
        print("✅ Using GCN...")
        gat_model = MultiModalGNN(
            feature_dim=config['GAT_dim'],
            gat_hidden_dim=512,
            gat_heads=4,
            num_modalities=num_modalities,
            fusion_type=fusion_type,
            bin_centers=bin_centers,
            use_res=False,
            classifier_arch=config.get('classifier_arch', None),
            use_gnn='GCN'
        ).to(device)
        config['use_GAT'] = True
        config['GNN_type'] = 'GCN'

    else:
        print("✅ Without GAT...")
        gat_model = MultiModalGNN(
            feature_dim=config['GAT_dim'],
            gat_hidden_dim=512,
            gat_heads=4,
            num_modalities=num_modalities,
            fusion_type=fusion_type,
            bin_centers=bin_centers,
            use_res=False,
            classifier_arch=config.get('classifier_arch', None)
        ).to(device)
        config['use_GAT'] = False
        config['GNN_type'] = None

    model.to(device)

    # --------------------------- Optimizer ---------------------------
    optimizer = optim.AdamW([
        {'params': model.parameters(), 'lr': config.get('lr', 1e-4) * 0.5},  # Feature extractor
        {'params': gat_model.parameters(), 'lr': config.get('lr', 1e-4)}     # GNN
    ])
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=1, eta_min=1e-7
    )
    scaler = GradScaler('cuda')  # Gradient scaler to prevent underflow in FP16

    # --------------------------- Loss Functions ---------------------------
    criterion = ClassificationRegressionLoss(alpha=config.get('alpha', 0.5), device=device)
    contrastive_criterion = ContrastiveLoss(config.get('temperature', 0.07)).to(device)

    # --------------------------- Metric Tracking ---------------------------
    metrics = {key: [] for key in [
        'train_loss', 'train_oa', 'train_r2', 'train_mae', 'train_maep',
        'val_loss', 'val_oa', 'val_r2', 'val_mae', 'val_maep'
    ]}
    best_val_loss = float('inf')

    # --------------------------- Training Loop ---------------------------
    model.train()
    gat_model.train()

    use_text = config.get('use_text', False)
    use_rsi = config.get('use_rsi', False)
    use_pois = config.get('use_pois', False)
    use_svis = config.get('use_svis', False)
    use_building = config.get('use_building', False)
    text_type = config.get('only_text', 'rsi')
    return_att = config.get('return_att', False)
    assert use_rsi or use_pois or use_building or use_text or use_svis, "At least one modality must be enabled"

    if config.get('CL_gamma', 0.1) > 0:
        use_CL = True 
    else:
        use_CL = False
        
    has_multimodal_pair = use_CL and use_text and (use_rsi or use_pois or use_svis or use_building)  # Whether to use contrastive loss
    if has_multimodal_pair:
        print(f"✅ Contrastive loss enabled with weight {config.get('CL_gamma', 0.1)}")

    rsis = data_batch['img_code']['rsi_code'].to(device) if use_rsi else None
    pois = data_batch['pois'].to(device) if use_pois else None
    svis = data_batch['svi'].to(device) if use_svis else None
    building_imgs = data_batch['img_code']['building_code'].to(device) if use_building else None
    text = {k: v.to(device) for k, v in data_batch['text'].items()} if use_text else None

    use_amp = config.get('use_amp', True)
    print(f"🚀 Mixed-precision training: {use_amp}")

    for epoch in range(config['epochs']):
        start_time = time.time()
        optimizer.zero_grad()  # Reset gradients

        # --- End-to-end forward pass: feature extraction + GNN propagation ---
        model.train()
        gat_model.train()

        # ----------- Forward Pass with Mixed Precision -----------
        with torch.amp.autocast('cuda', dtype=torch.float16, enabled=use_amp):  # Enable AMP acceleration
            # Feature extractor forward
            modal_features = model(rsis, pois, building_imgs, svis, text, only_feature=True)

            # Define modality pairs for contrastive learning
            contrastive_pairs = []
            if config.get('contrastive_rsi_text', True) and use_rsi and use_text:
                contrastive_pairs.append(('rsi', 'rsi_text'))
            if config.get('contrastive_poi_text', True) and use_pois and use_text:
                contrastive_pairs.append(('poi', 'poi_text'))
            if config.get('contrastive_building_text', True) and use_building and use_text:
                contrastive_pairs.append(('building', 'building_text'))
            if config.get('contrastive_svi_text', True) and use_svis and use_text:
                contrastive_pairs.append(('svi', 'svi_text'))

            # x: [N, C], rsi_feats_all: [N, C], both on GPU with gradients!

            # GNN forward
            x_gnn, C_m, _ = gat_model(modal_features, data, return_contrastive_loss=use_CL, return_embeddings=False, return_score=return_att, use_GNN=config['use_GAT'])
            logits = x_gnn  # [N, 24, num_classes]

            # Full-graph logits: [N, 24, C]
            N, T, C = logits.shape

            # Use train_mask to select nodes
            train_logits = logits[data.train_mask]        # [3157, 24, C]
            train_labels = data.y_class[data.train_mask]  # [3157, 24]
            train_y_reg = data.y_reg[data.train_mask]     # [3157, 24]

            # Flatten
            train_logits_flat = train_logits.reshape(-1, C)     # [3157*24, C]
            train_labels_flat = train_labels.reshape(-1)        # [3157*24]
            train_y_reg_flat = train_y_reg.reshape(-1)          # [3157*24]

            # Regression prediction
            probs_flat = torch.softmax(train_logits_flat, dim=1)
            train_reg_preds_flat = torch.sum(probs_flat * bin_centers, dim=1)  # [3157*24]

            # Main loss
            main_loss = criterion(train_logits_flat, train_reg_preds_flat, train_labels_flat, train_y_reg_flat)

            # Image-text contrastive loss (training set only)
            # Feature extraction on full graph → contrastive loss only on training nodes
            contrastive_loss_list = []
            for modal1, modal2 in contrastive_pairs:
                if modal1 in modal_features and modal2 in modal_features:
                    # Ensure feature dimensions match
                    geo_feat = modal_features[modal1][data.train_mask]
                    txt_feat = modal_features[modal2][data.train_mask]

                    loss_pair = contrastive_criterion(geo_feat, txt_feat)
                    contrastive_loss_list.append(loss_pair * config.get('CL_gamma', 0.1))

            if contrastive_loss_list:
                contrastive_loss = torch.stack(contrastive_loss_list).mean()
            else:
                contrastive_loss = torch.tensor(0.0, device=device, requires_grad=True)

            contrastive_loss = contrastive_loss + C_m * config.get('CL_gamma2', 0.1)
            loss = main_loss + contrastive_loss

        # --- Backward pass ---
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Optimizer step
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        # Learning rate scheduler
        scheduler.step(epoch)

        # --- Training Metrics ---
        train_loss = loss.item()

        # Classification accuracy OA (Overall Accuracy)
        pred_labels_flat = torch.argmax(train_logits_flat, dim=-1)  # [3157*24]
        train_oa = (pred_labels_flat == train_labels_flat).float().mean().item()

        # Regression metrics
        y_true_t = train_y_reg_flat.cpu().numpy()           # [3157*24]
        y_pred_t = train_reg_preds_flat.detach().cpu().numpy()    # [3157*24]

        
        y_true_filtered = (processor.inverse_transform(y_true_t))      # Inverse transform & filter
        y_pred_filtered = (processor.inverse_transform(y_pred_t))

        y_true_denorm = processor.inverse_transform(y_true_t, type='original')
        y_pred_denorm = processor.inverse_transform(y_pred_t, type='original')

        train_r2 = r2_score(y_true_t, y_pred_t)
        train_mae = mean_absolute_error(y_true_denorm, y_pred_denorm)

        mask_zero = y_true_filtered > 0.1         # Filter out samples with true value == 0 
        y_true_filtered = y_true_filtered[mask_zero]
        y_pred_filtered = y_pred_filtered[mask_zero]

        # Compute MAPE: Mean Absolute Percentage Error
        train_maep = np.mean(np.abs(y_pred_filtered - y_true_filtered) / (y_true_filtered)) * 100

        # --- Validation Phase ---
        model.eval()
        gat_model.eval()
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.float16, enabled=use_amp):  # Enable AMP for inference
                # Forward pass during evaluation
                modal_features = model(rsis, pois, building_imgs, svis, text, only_feature=True)

                # GNN forward (full graph)
                val_x_gnn, C_L, modalities_dict = gat_model(modal_features, data, return_contrastive_loss=False, return_embeddings=True, use_GNN=config['use_GAT'])
                val_logits = val_x_gnn  # [N, 24, num_classes]

                # Extract validation nodes
                val_mask = data.val_mask
                val_logits_node = val_logits[val_mask]        # [N_val, 24, num_classes]
                val_labels_node = data.y_class[val_mask]      # [N_val, 24]
                val_y_reg_node = data.y_reg[val_mask]         # [N_val, 24]

                # Flatten
                B_val, T_val, C_val = val_logits_node.shape
                val_logits_flat = val_logits_node.reshape(B_val * T_val, C_val)     # [N_val*24, C]
                val_labels_flat = val_labels_node.reshape(-1)                       # [N_val*24]
                val_y_reg_flat = val_y_reg_node.reshape(-1)                         # [N_val*24]

                # Regression prediction
                val_probs = torch.softmax(val_logits_flat, dim=1)
                val_reg_preds_flat = torch.sum(val_probs * bin_centers, dim=1)      # [N_val*24]

                # ✅ Validation loss: computed only on validation nodes
                val_loss = criterion(
                    class_logits=val_logits_flat,
                    reg_preds=val_reg_preds_flat,
                    class_labels=val_labels_flat,
                    reg_targets=val_y_reg_flat
                )
                val_loss = (val_loss + C_L).item()

            # Classification accuracy OA
            pred_labels_flat = torch.argmax(val_logits_flat, dim=-1)
            val_oa = (pred_labels_flat == val_labels_flat).float().mean().item()

            # Regression metrics
            y_true_v = val_y_reg_flat.cpu().numpy()
            y_pred_v = val_reg_preds_flat.cpu().numpy() 

            y_true_denorm = processor.inverse_transform(y_true_t, type='original')
            y_pred_denorm = processor.inverse_transform(y_pred_t, type='original')
        
            val_r2 = r2_score(y_true_v, y_pred_v)
            val_mae = mean_absolute_error(y_true_denorm, y_pred_denorm)
            
            # Filter out samples with true value == 0
            y_true_filtered = (processor.inverse_transform(y_true_v))     # Inverse transform & filter
            y_pred_filtered = (processor.inverse_transform(y_pred_v))     
            mask_zero = y_true_filtered > 0.1                 # Filter out zero-valued samples
            y_true_filtered = y_true_filtered[mask_zero]
            y_pred_filtered = y_pred_filtered[mask_zero]

            # Compute MAPE
            val_maep = np.mean(np.abs(y_pred_filtered - y_true_filtered) / (y_true_filtered)) * 100

        # --- Logging ---
        metrics['train_loss'].append(train_loss)
        metrics['train_oa'].append(train_oa)
        metrics['train_r2'].append(train_r2)
        metrics['train_mae'].append(train_mae)
        metrics['train_maep'].append(train_maep)

        metrics['val_loss'].append(val_loss)
        metrics['val_oa'].append(val_oa)
        metrics['val_r2'].append(val_r2)
        metrics['val_mae'].append(val_mae)
        metrics['val_maep'].append(val_maep)

        # --- Save Best Model + Cache Predictions ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            # Cache predictions
            with torch.no_grad():
                train_probs = torch.softmax(logits, dim=-1).cpu().numpy()  # [N, 24, num_classes]
                val_probs = torch.softmax(val_logits, dim=-1).cpu().numpy()
                labels_np = data.y_class.cpu().numpy()  # [N, 24]

            # Save as list of arrays (compatible with save_confusion_matrix_and_metrics interface)
            best_train_preds = [train_probs[data.train_mask.cpu().numpy()]]
            best_train_targets = [labels_np[data.train_mask.cpu().numpy()]]
            best_val_preds = [val_probs[data.val_mask.cpu().numpy()]]
            best_val_targets = [labels_np[data.val_mask.cpu().numpy()]]

            # Save model
            if config.get('save_path', None):
                torch.save({
                    'model': model.state_dict(),
                    'gat_model': gat_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'config': config,
                    'metrics': metrics,
                    'epoch': epoch,
                }, config['save_path'])
            else:
                print("⚠️ Warning: No save path configured, skipping model saving")

        # --- Print Progress ---
        if (epoch + 1) % 5 == 0:
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1:2d} | Time: {epoch_time:.2f}s | "
                f"Train Loss: {train_loss:.4f} | CL: {contrastive_loss:.4f} | OA: {train_oa:.4f} | R²: {train_r2:.4f} | MAE: {train_mae:.4f} | MAPE: {train_maep:.2f}% | "
                f"Val Loss: {val_loss:.4f} | OA: {val_oa:.4f} | R²: {val_r2:.4f} | MAE: {val_mae:.4f} | MAPE: {val_maep:.2f}%")
            
        # Visualize embeddings
        if (epoch + 1) % 60 == 0 and config.get('visualize_embeddings', False):
            save_path = os.path.join(r'/workspace/data/program/visualize_image/t-SNE', f"{config['fusion_type']}: tsne_epoch_{epoch + 1}.png")
            plot_tsne_embeddings(
                modalities_dict=modalities_dict,
                save_path=save_path,
                title=f"{config['fusion_type']}: t-SNE at Epoch {epoch + 1}",
                perp=20
            )

    # --------------------------- Save Metrics and Confusion Matrix ---------------------------
    print(f"✅ Best model saved to {config['save_path']}")
    np.savez(config['metrics_save_path'], **metrics)
    if config.get('save_path', None):
        print(f"📊 Training metrics saved to {config['metrics_save_path']}")
    else:
        print("⚠️ Warning: No save path configured, skipping metrics saving")

    # Save confusion matrix if path is provided
    if 'confusion_matrix_path' in config:
        os.makedirs(os.path.dirname(config['confusion_matrix_path']), exist_ok=True)
        
        # Ensure best predictions are cached
        if best_train_preds is not None and best_val_preds is not None:
            save_confusion_matrix_and_metrics(
                config=config,
                all_train_preds=best_train_preds,
                all_train_targets=best_train_targets,
                all_val_preds=best_val_preds,
                all_val_targets=best_val_targets
            )
            print(f"📊 Confusion matrix saved to {config['confusion_matrix_path']}")
        else:
            print("⚠️ Warning: No best predictions cached, skipping confusion matrix saving")

    print(f"📈 Best validation OA: {np.max(metrics['val_oa']):.4f}, Best validation R²: {np.max(metrics['val_r2']):.4f}, "
          f"Best validation MAE: {np.min(metrics['val_mae']):.4f}, Best validation MAPE: {np.min(metrics['val_maep']):.2f}%")
    return model


def save_confusion_matrix_and_metrics(config, all_train_preds, all_train_targets, all_val_preds, all_val_targets):
    """
    Compute and save confusion matrices and classification metrics (Precision, Recall, F1).

    Args:
        config: Configuration dictionary containing 'confusion_matrix_path'
        all_train_preds: list of np.ndarray, each of shape [batch_size, 24, num_classes]
        all_train_targets: list of np.ndarray, each of shape [batch_size, 24]
        all_val_preds: list of np.ndarray
        all_val_targets: list of np.ndarray
    """
    
    # Concatenate all batches
    train_preds = np.concatenate(all_train_preds)    # [N_train, 24, num_classes]
    train_targets = np.concatenate(all_train_targets) # [N_train, 24]
    val_preds = np.concatenate(all_val_preds)        # [N_val, 24, num_classes]
    val_targets = np.concatenate(all_val_targets)    # [N_val, 24]

    # Flatten across time steps and samples
    num_classes = train_preds.shape[-1]
    train_preds_flat = np.argmax(train_preds.reshape(-1, num_classes), axis=1)  # [N_train * 24]
    train_targets_flat = train_targets.reshape(-1)                              # [N_train * 24]

    val_preds_flat = np.argmax(val_preds.reshape(-1, num_classes), axis=1)
    val_targets_flat = val_targets.reshape(-1)

    # Generate confusion matrices
    train_cm = confusion_matrix(train_targets_flat, train_preds_flat)
    val_cm = confusion_matrix(val_targets_flat, val_preds_flat)

    # Save
    np.savez(config['confusion_matrix_path'], train_cm=train_cm, val_cm=val_cm)
    print(f"✅ Confusion matrices saved to {config['confusion_matrix_path']}")

    # Compute metrics
    avg_mode = 'weighted'  # Can be changed to 'macro' or 'micro'

    def safe_score(func, y_true, y_pred, **kwargs):
        try:
            return func(y_true, y_pred, **kwargs)
        except Exception as e:
            print(f"⚠️ Error computing {func.__name__}: {e}")
            return np.nan

    precision = safe_score(precision_score, train_targets_flat, train_preds_flat, average=avg_mode)
    recall = safe_score(recall_score, train_targets_flat, train_preds_flat, average=avg_mode)
    f1 = safe_score(f1_score, train_targets_flat, train_preds_flat, average=avg_mode)

    precision_val = safe_score(precision_score, val_targets_flat, val_preds_flat, average=avg_mode)
    recall_val = safe_score(recall_score, val_targets_flat, val_preds_flat, average=avg_mode)
    f1_val = safe_score(f1_score, val_targets_flat, val_preds_flat, average=avg_mode)

    # Print results
    print(f"📊 Training Set Metrics ({avg_mode}-average):")
    print(f"   Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}")

    print(f"📊 Validation Set Metrics ({avg_mode}-average):")
    print(f"   Precision: {precision_val:.4f} | Recall: {recall_val:.4f} | F1 Score: {f1_val:.4f}")
    print()
