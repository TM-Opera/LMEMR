import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
from itertools import combinations
from uni_model import MultimodalFusion
import numpy as np
from Custom_classifier import SelfClassifier

# ----------------- Phase 2: Graph Neural Network + Temporal Modeling -----------------

# Graph Attention Residual Block
class GATResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=4, dropout=0.2, use_res=True, concat=True):
        super().__init__()
        self.conv = GATConv(in_channels, out_channels, heads=num_heads, dropout=dropout, concat=concat)
        self.norm = nn.LayerNorm(in_channels)
        self.dropout = dropout
        if use_res:
            # Residual connection: linear layer if dimensions don't match
            self.residual = nn.Linear(in_channels, out_channels * num_heads) if in_channels != out_channels * num_heads else None
        else:
            self.residual = None

        self.reset_parameters()

    # Parameter initialization
    def reset_parameters(self):
        self.conv.reset_parameters()
        if self.residual is not None:
            nn.init.xavier_uniform_(self.residual.weight)
            if self.residual.bias is not None:
                nn.init.zeros_(self.residual.bias)

    def forward(self, x, edge_index, return_attention=False):
        res = x                        # Store residual
        x = self.norm(x)               # Pre-Layer Normalization
    
        if return_attention:
            # Forward with attention weights
            x, (edge_idx_att, att_weights) = self.conv(x, edge_index, return_attention_weights=True)
        else:
            x = self.conv(x, edge_index)

        # Handle unexpected tuple output from GATConv
        if isinstance(x, tuple):
            print("GATConv returned a tuple, taking the first element.")
            print(f'flag: {len(x)}, {return_attention}')
            x = x[0]  
            
        x = F.gelu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Apply residual connection
        if self.residual is not None:
            res = self.residual(res)
            x = x + res

        if return_attention:
            return x, att_weights.detach()  # Return raw attention weights
        else:
            return x

# =====================================================
# Step 1: Define a 3-layer GAT or GCN network
# =====================================================
class GATNetwork(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=4, dropout=0.2, use_res=True):
        super(GATNetwork, self).__init__()
        self.dropout = dropout

        # Three GATResBlocks with increasing depth
        self.conv1 = GATResBlock(in_channels, hidden_channels, num_heads=num_heads, dropout=dropout, concat=True, use_res=use_res)
        self.conv2 = GATResBlock(hidden_channels * num_heads, hidden_channels, num_heads=num_heads, dropout=dropout, concat=True, use_res=use_res)
        self.conv3 = GATResBlock(hidden_channels * num_heads, out_channels, num_heads=1, dropout=dropout, concat=False, use_res=use_res)

    def forward(self, x, edge_index, return_attention=False):
        # Forward through three GAT layers
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)

        if return_attention:
            x, att_weights = self.conv3(x, edge_index, return_attention=True)
            return x, att_weights  # [N, out_channels], [E, 1]
        else:
            x = self.conv3(x, edge_index, return_attention=False)
            return x, torch.tensor(0.0)  # Placeholder for consistency

class GCNNetwork(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=4, dropout=0.2, use_res=True):
        super(GCNNetwork, self).__init__()
        self.dropout = dropout

        # Use GCNConv instead of GATResBlock
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)

        self.norm = nn.LayerNorm(in_channels)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x, edge_index, return_attention=False):
        x = self.norm(x)  # Pre-Normalization

        # First GCN layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout_layer(x)

        # Second GCN layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout_layer(x)

        # Third GCN layer
        if return_attention:
            # GCN does not support attention; return placeholder
            x = self.conv3(x, edge_index)
            att_weights = torch.zeros(edge_index.size(1), 1, device=x.device)  # [E, 1] placeholder
            return x, att_weights
        else:
            x = self.conv3(x, edge_index)
            return x, torch.tensor(0.0)  # Maintain consistent output format

# =====================================================
# Step 2: Define multimodal contrastive learning module
# ===================================================== 
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, feat1, feat2):
        """
        Compute symmetric contrastive loss between two modalities.

        Args:
            feat1: Tensor of shape [B, C]
            feat2: Tensor of shape [B, C]

        Returns:
            loss: scalar tensor
        """
        device = feat1.device
        batch_size = feat1.shape[0]

        # L2 normalize features
        feat1 = F.normalize(feat1, dim=1)
        feat2 = F.normalize(feat2, dim=1)

        # Similarity matrix
        sim = torch.matmul(feat1, feat2.t()) / self.temperature  # [B, B]

        # Labels: diagonal elements are positive pairs
        labels = torch.arange(batch_size).to(device)

        # Symmetric loss: both directions
        loss_1 = F.cross_entropy(sim, labels)
        loss_2 = F.cross_entropy(sim.t(), labels)

        loss = (loss_1 + loss_2) / 2 

        return loss       

# =====================================================
# Step 3: Define multimodal GNN network
# =====================================================
class MultiModalGNN(nn.Module):
    def __init__(self, feature_dim=256, gat_hidden_dim=512, gat_heads=4, num_modalities=4, fusion_type='concat', bin_centers=None, use_res=True, classifier_arch=None, use_gnn='GAT'):
        super(MultiModalGNN, self).__init__()

        # GNNs for each modality
        if use_gnn == 'GAT':
            self.gnns = nn.ModuleDict({
                'rsi': GATNetwork(in_channels=feature_dim, hidden_channels=gat_hidden_dim, out_channels=feature_dim, num_heads=gat_heads, use_res=use_res),
                'poi': GATNetwork(in_channels=feature_dim, hidden_channels=gat_hidden_dim, out_channels=feature_dim, num_heads=gat_heads, use_res=use_res),
                'building': GATNetwork(in_channels=feature_dim, hidden_channels=gat_hidden_dim, out_channels=feature_dim, num_heads=gat_heads, use_res=use_res),
                'svi': GATNetwork(in_channels=feature_dim, hidden_channels=gat_hidden_dim, out_channels=feature_dim, num_heads=gat_heads, use_res=use_res),
            })
        else:
            self.gnns = nn.ModuleDict({
                'rsi': GCNNetwork(in_channels=feature_dim, hidden_channels=gat_hidden_dim, out_channels=feature_dim, num_heads=gat_heads, use_res=use_res),
                'poi': GCNNetwork(in_channels=feature_dim, hidden_channels=gat_hidden_dim, out_channels=feature_dim, num_heads=gat_heads, use_res=use_res),
                'building': GCNNetwork(in_channels=feature_dim, hidden_channels=gat_hidden_dim, out_channels=feature_dim, num_heads=gat_heads, use_res=use_res),
                'svi': GCNNetwork(in_channels=feature_dim, hidden_channels=gat_hidden_dim, out_channels=feature_dim, num_heads=gat_heads, use_res=use_res),
            })
    
        # Fusion module
        self.fusion_net = MultimodalFusion(feature_dim=feature_dim, num_modalities=num_modalities, fusion_type=fusion_type)
        
        # Classification head
        self.classifier = SelfClassifier(input_dim=feature_dim, arch=classifier_arch)

        # Contrastive loss module for cross-modal alignment
        self.contrast_loss = ContrastiveLoss()

    def forward(self, data_dict, data, return_contrastive_loss=True, return_embeddings=False, use_GNN=True, return_score=False):
        """
        Forward pass for multimodal GNN.

        Args:
            data_dict: dict containing modality features, keys in ['rsi', 'poi', 'building', 'svi']
                       values can be tensors or None
            data: graph structure (e.g., edge index)
            return_contrastive_loss: bool, whether to compute and return contrastive loss
            use_GNN: bool, whether to apply GNN encoding
            return_score: bool, whether to return attention scores

        Returns:
            logits: classification output
            total_contrastive_loss: average contrastive loss over valid modality pairs (scalar tensor)
            modalities: optional, dictionary of encoded modality features (if return_embeddings=True)
        """
        edge_index = data.edge_index
        device = edge_index.device if isinstance(edge_index, torch.Tensor) else next(self.parameters()).device

        # Store valid modality features
        valid_features = []
        att_dict = {}

        modalities = {}
        for key in ['rsi', 'poi', 'building', 'svi']:
            feat = data_dict.get(key, None)
            if feat is None or feat.shape[1] == 0:  
                continue

            # Ensure feature is on correct device
            if isinstance(feat, torch.Tensor):
                feat = feat.to(device)
            else:
                continue  # Skip non-tensor inputs

            # Encode via corresponding GNN
            if use_GNN:
                encoded_feat, att_weights = self.gnns[key](feat, edge_index, return_score)
                att_dict[key] = att_weights.cpu().numpy()
            else:
                encoded_feat = feat
            modalities[key] = encoded_feat

        if len(modalities) == 0:
            raise ValueError("No valid modality found in input data_dict.")
        
        # Save attention weights if all four modalities exist and requested
        if len(att_dict) == 4 and return_score:
            att_dict['edge_index'] = edge_index.cpu().numpy()
            np.savez('GAT_Att_Weights/attention_weights_all_modalities.npz', **att_dict)
            print("✅ Attention dictionary saved to 'GAT_Att_Weights/attention_weights_all_modalities.npz'")

        # Fuse features from available modalities
        fused_features = self.fusion_net(list(modalities.values()))
        logits = self.classifier(fused_features)

        if not return_contrastive_loss:
            if return_embeddings:
                return logits, torch.tensor(0.0, device=device, requires_grad=True), modalities
            else:
                return logits, torch.tensor(0.0, device=device, requires_grad=True), None

        # === Compute contrastive loss only for existing modality pairs ===
        if len(modalities) < 2:
            # Cannot compute contrastive loss with fewer than 2 modalities
            total_contrastive_loss = torch.tensor(0.0, device=device, requires_grad=True)
        else:
            contrastive_losses = []
            mod_names = list(modalities.keys())
            # Iterate over all combinations of modality pairs
            for i, j in combinations(range(len(mod_names)), 2):
                mod1, mod2 = mod_names[i], mod_names[j]
                feat1, feat2 = modalities[mod1], modalities[mod2]
                loss = self.contrast_loss(feat1, feat2)
                contrastive_losses.append(loss)

            # Average loss across all valid modality pairs
            total_contrastive_loss = torch.stack(contrastive_losses).mean()

        return logits, total_contrastive_loss, None