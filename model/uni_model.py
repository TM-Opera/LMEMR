import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

# ------------------ Submodule Definitions ------------------

class MLP(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=1024, output_dim=768):
        super().__init__()
       
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)


class POIEncoder(nn.Module):
    def __init__(self, output_dim=256):  
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(23, output_dim // 2),  
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(output_dim // 2, output_dim),  
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(output_dim, 4 * output_dim),  
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(4 * output_dim, output_dim)
        )
    
    def forward(self, x):
        if x.dim() == 1:  # If input is (23,), reshape to (1, 23)
            x = x.unsqueeze(0)
        assert x.shape[1] == 23, f"Input shape should be (batch_size, 23), got {x.shape}"
        return self.model(x)  # Output shape: (batch_size, 256)
    

class SVIEncoder(nn.Module):
    def __init__(self, output_dim=512):
        super(SVIEncoder, self).__init__()
        self.proj = nn.Linear(2048, 128)
        self.model = nn.Sequential(
            nn.Linear(128*20, 1024),  
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(1024, 1024),  
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(1024, 1024),   
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(1024, output_dim)
        )
    
    def forward(self, x):
        """
        Input x: (B, 20, 512) or single sample (20, 512)
        Output: (B, 20 * 128)
        """
        B, N, C = x.shape  # B=batch size, N=20
        assert N == 20, f"Expected 20 images, got {N}"

        # Feature projection: (B, N, 512) → (B, N, 128)
        features = self.proj(x)  # Output (B, 20, 128)

        # Reshape to (B, 20*128)
        features = features.reshape(B, -1)   # (B, 20*128)

        return self.model(features)  # (B, 2560) -> (B, d_model)

    

class CrossModalityAlignment(nn.Module):
    """
    Cross-modality alignment module based on channel-wise similarity retrieval.
    Aligns two modalities (e.g., visual and textual) by learning attention-like weights via channel correlation.
    """
    def __init__(self, d_model=512):
        super().__init__()
        self.psi_q = nn.Linear(d_model, d_model, bias=True)  # Processes query embedding (e.g., image)
        self.psi_v = nn.Linear(d_model, d_model, bias=True)  # Processes value of key modality (e.g., text)
        self.psi_k = nn.Linear(d_model, d_model, bias=True)  # Processes key of key modality
        self.omega_c = nn.Linear(d_model, d_model, bias=True)  # Final linear transformation

        self.d_model = d_model    

    def forward(self, Q_emb, K_emb):
        """
        Forward pass: fuse two embeddings based on channel similarity.

        Args:
            Q_emb: Query embedding from primary modality (e.g., CV features), shape [batch_size, C]
            K_emb: Key embedding from auxiliary modality (e.g., text prompts), shape [batch_size, C]

        Returns:
            aligned_emb: Aligned fused embedding, shape [batch_size, d_model]
        """
        q = self.psi_q(Q_emb).transpose(0, 1)  # [C, batch_size]
        k = self.psi_k(K_emb)  # [batch_size, C]
        v = self.psi_v(K_emb).transpose(0, 1)  # [C, batch_size]
        
        # Compute channel similarity matrix: [C, C]
        similarity = torch.matmul(q, k)   # [C, C]
        similarity_matrix = F.softmax(similarity, dim=-1)  # Normalize along last dimension
        
        # Aggregation: [C, C] @ [C, batch_size] -> [C, batch_size]
        aggregated = torch.matmul(similarity_matrix, v).transpose(0, 1)  # [batch_size, C]
        aggregated_f = self.omega_c(aggregated)  # [batch_size, d_model]

        # Residual connection
        aligned_emb = aggregated_f + Q_emb     # [batch_size, d_model]
        
        return aligned_emb
    

class GatedFusion(nn.Module):
    """
    Simple gated fusion model supporting arbitrary number of input modalities.

    Args:
        feature_dim (int): Feature dimension per modality (assumes all have same dimension)
        num_modalities (int): Number of input modalities
    """
    def __init__(self, feature_dim, num_modalities):
        super(GatedFusion, self).__init__()
        self.feature_dim = feature_dim
        self.num_modalities = num_modalities
        
        # Gating network: maps concatenated features to weights for each modality
        self.gate_network = nn.Linear(feature_dim * num_modalities, num_modalities)
        
    def forward(self, modalities):
        """
        Forward pass.

        Args:
            modalities: List of tensors, each with shape (B, D), total length = num_modalities
                B: batch size, D: feature_dim

        Returns:
            fused: Fused feature tensor, shape (B, D)
            weights: Gating weights (softmax normalized), shape (B, num_modalities)
        """
        assert len(modalities) == self.num_modalities, \
            f"Expected {self.num_modalities} modalities, got {len(modalities)}"
        
        # Concatenate all modalities: (B, D * M)
        concatenated = torch.cat(modalities, dim=1)  # (B, D*M)
        
        # Generate gating weights: (B, M)
        gate_weights = self.gate_network(concatenated)  # (B, M)
        gate_weights = F.softmax(gate_weights, dim=1)  # Normalize to probabilities
        
        # Weighted fusion: multiply each modality by its weight and sum
        weighted_features = []
        for i, mod in enumerate(modalities):
            weight = gate_weights[:, i].unsqueeze(1)  # (B, 1)
            weighted_features.append(weight * mod)    # (B, D)
        
        fused = sum(weighted_features)  # (B, D)
        
        return fused, gate_weights


class MultimodalFusion(nn.Module):
    def __init__(self, 
                 feature_dim: int, 
                 num_modalities: int, 
                 num_heads: int = 2, 
                 num_layers: int = 1, 
                 dropout: float = 0.2, 
                 fusion_type: str = 'concat'):
        """
        Multimodal fusion module with various fusion strategies.

        Args:
            feature_dim: Feature dimension D for each modality
            num_modalities: Number of modalities N
            num_heads: Number of attention heads (for 'attention' fusion)
            num_layers: Number of Transformer layers (for 'attention')
            dropout: Dropout probability
            fusion_type: Fusion strategy: 'attention' | 'concat' | 'maxpooling' | 'gated'
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.num_modalities = num_modalities
        self.fusion_type = fusion_type

        # ✅ Modality type embedding to distinguish different modalities
        self.modality_type_embedding = nn.Embedding(num_modalities, feature_dim)
        nn.init.normal_(self.modality_type_embedding.weight, std=0.02)

        # ========== Projection heads: align modalities into shared space ==========
        self.proj_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim, feature_dim)
            ) for _ in range(num_modalities)
        ])

        # ========== Fusion architecture definition ==========
        if fusion_type == 'attention':
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=feature_dim,
                nhead=num_heads,
                dim_feedforward=feature_dim * 4,
                dropout=dropout,
                batch_first=True,
                norm_first=True  # Pre-LayerNorm
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
            self.fuse_layer = nn.Linear(feature_dim * num_modalities, feature_dim)
            self.output_proj = nn.Linear(feature_dim, feature_dim)

        elif fusion_type == 'concat':
            self.output_proj = nn.Sequential(
                nn.Linear(feature_dim * num_modalities, feature_dim * num_modalities),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(feature_dim * num_modalities, feature_dim)
            )

        elif fusion_type == 'maxpooling':
            self.output_proj = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(feature_dim, feature_dim)
            )
        elif fusion_type == 'gated':
            self.gate_network = nn.Linear(feature_dim * num_modalities, num_modalities)
        else:
            raise ValueError(f"Unsupported fusion_type: {fusion_type}")

    def forward(self, modality_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass.

        Args:
            modality_features: List of tensors, each shape [B, D], length = N

        Returns:
            fused feature [B, D]
        """
        B = modality_features[0].shape[0]

        # ========== Step 1: Project each modality and add type embedding ==========
        projected_features = []
        for i, feat in enumerate(modality_features):
            feat = self.proj_heads[i](feat)  # [B, D]
            feat = feat + self.modality_type_embedding.weight[i]  # ✅ Add modality type embedding
            projected_features.append(feat)

        # ========== Step 2: Fuse according to fusion_type ==========
        if self.fusion_type == 'attention':
            x = torch.stack(projected_features, dim=1)  # [B, N, D]
            x = self.transformer(x)  # [B, N, D]
            fused = self.fuse_layer(x.reshape(B, -1))  # Flatten and project
            fused = self.output_proj(fused)

        elif self.fusion_type == 'concat':
            x = torch.cat(projected_features, dim=-1)  # [B, N*D]
            fused = self.output_proj(x)  # [B, D]

        elif self.fusion_type == 'maxpooling':
            x = torch.stack(projected_features, dim=1)  # [B, N, D]
            x = x.max(dim=1)[0]  # [B, D], max pooling across modalities
            fused = self.output_proj(x)  # [B, D]

        elif self.fusion_type == 'gated':
            x = torch.cat(projected_features, dim=-1)  # [B, N*D]
            gate_weights = self.gate_network(x)  # (B, M)
            gate_weights = F.softmax(gate_weights, dim=1)
            weighted_features = []
            for i, mod in enumerate(projected_features):
                weight = gate_weights[:, i].unsqueeze(1)  # (B, 1)
                weighted_features.append(weight * mod)    # (B, D)
            fused = sum(weighted_features)  # (B, D)

        else:
            raise ValueError(f"Unsupported fusion_type: {self.fusion_type}")

        return fused