import torch
import torch.nn as nn
from uni_model import POIEncoder, SVIEncoder, MLP, CrossModalityAlignment

# ------------------ Phase 1: Feature Extraction + Semantic Fusion ------------------

# --------------------------
# Multimodal Semantic Enhancer
# --------------------------
class Multimodal_Semantic_Enhancer(nn.Module):
    def __init__(self, feature_dim=256, fusion_type='concat', fusion_num=1, text_list=[]):
        """
        Initialize the multimodal semantic enhancer.

        Args:
            feature_dim: dimension of extracted features
            fusion_type: type of fusion strategy (e.g., 'concat')
            fusion_num: number of fusion layers or blocks
            text_list: list of text modality names (e.g., ['rsi', 'poi', 'building', 'svi'])
        """
        super().__init__()
        self.fusion_num = fusion_num
        self.text_list = text_list

        # Remote sensing image feature extractor
        self.img_feature_extractor = MLP(in_dim=2048, hidden_dim=4096, output_dim=feature_dim)

        # Building height image feature extractor
        self.building_feature_extractor = MLP(in_dim=2048, hidden_dim=4096, output_dim=feature_dim)

        # POI feature extractor
        self.poi_feature_extractor = POIEncoder(output_dim=feature_dim)

        # SVI (Street View Image) feature extractor
        self.svi_feature_extractor = SVIEncoder(output_dim=feature_dim)

        # Text feature extractors (one MLP per modality)
        self.prompt_feature_extractor = nn.ModuleList([
            MLP(in_dim=4096, hidden_dim=4096, output_dim=feature_dim)
            for _ in range(4)
        ])

        # Cross-modality alignment modules (one per modality pair)
        self.cross_modality_alignment = nn.ModuleList([
            CrossModalityAlignment(d_model=feature_dim)
            for _ in range(4)
        ])

        # Optional regressor (to directly output flow values if needed)
        self.regressor = nn.Linear(feature_dim, 24)

    def forward(self, imgs=None, pois=None, building_imgs=None, svis=None, text=None, only_feature=True):
        """
        Forward pass through the model.

        Args:
            imgs: remote sensing images [batch_size, 3, height, width]
            pois: POI features [batch_size, poi_dim]
            building_imgs: building height images [batch_size, 1, height, width]
            svis: street view image features [batch_size, svi_dim * svi_num]
            text: dictionary of text embeddings, e.g., {'rsi': ..., 'poi': ...}
            only_feature: if True, only return fused features; otherwise may return additional outputs

        Returns:
            If only_feature=True:
                modal_features: dict containing aligned features for each modality
            Otherwise:
                Returns classification and/or regression outputs (not implemented in this snippet)
        """
        # Automatically expand dimensions if inputs are missing batch dimension
        if imgs is not None and imgs.dim() == 3:
            imgs = imgs.unsqueeze(0)  # [C,H,W] -> [1,C,H,W]
        if building_imgs is not None and building_imgs.dim() == 3:
            building_imgs = building_imgs.unsqueeze(0)  # [C,H,W] -> [1,C,H,W]    
            
        # Check presence and validity of each input modality
        has_imgs = imgs is not None and imgs.numel() > 0
        has_pois = pois is not None and pois.numel() > 0
        has_building = building_imgs is not None and building_imgs.numel() > 0
        has_svis = svis is not None and svis.numel() > 0
        has_text = text is not None

        # Ensure at least one modality is provided
        if not (has_imgs or has_pois or has_building or has_text or has_svis):
            raise ValueError("At least one of RSI, POI, SVI, Building, or text features must be provided")

        # List to collect features for fusion
        features_list = []
        modal_features = {}  # Dictionary to store final aligned features per modality
        text_list = self.text_list

        # Process RS Image modality
        if has_imgs:
            img_features = self.img_feature_extractor(imgs)
            if has_text:
                text_features = self.prompt_feature_extractor[0](text['rsi'])
                aligned_emb = self.cross_modality_alignment[0](img_features, text_features)
            else:
                text_features = torch.zeros_like(img_features)
                aligned_emb = self.cross_modality_alignment[0](img_features, img_features)
            features_list.append(aligned_emb)
            modal_features['rsi'] = aligned_emb
            modal_features['rsi_text'] = text_features  # Store corresponding text features

        # Process POI modality
        if has_pois:
            poi_features = self.poi_feature_extractor(pois)
            if has_text:
                text_features = self.prompt_feature_extractor[1](text['poi'])
                aligned_emb = self.cross_modality_alignment[1](poi_features, text_features)
            else:
                text_features = torch.zeros_like(poi_features)
                aligned_emb = self.cross_modality_alignment[1](poi_features, poi_features)
            features_list.append(aligned_emb)
            modal_features['poi'] = aligned_emb
            modal_features['poi_text'] = text_features

        # Process Building Height modality
        if has_building:
            building_features = self.building_feature_extractor(building_imgs)
            if has_text:
                text_features = self.prompt_feature_extractor[2](text['building'])
                aligned_emb = self.cross_modality_alignment[2](building_features, text_features)
            else:
                text_features = torch.zeros_like(building_features)
                aligned_emb = self.cross_modality_alignment[2](building_features, building_features)
            features_list.append(aligned_emb)
            modal_features['building'] = aligned_emb
            modal_features['building_text'] = text_features

        # Process SVI (Street View) modality
        if has_svis:
            svis_features = self.svi_feature_extractor(svis)
            if has_text:
                text_features = self.prompt_feature_extractor[3](text['svi'])
                aligned_emb = self.cross_modality_alignment[3](svis_features, text_features)
            else:
                text_features = torch.zeros_like(svis_features)
                aligned_emb = self.cross_modality_alignment[3](svis_features, svis_features)
            features_list.append(aligned_emb)
            modal_features['svi'] = aligned_emb
            modal_features['svi_text'] = text_features

        # Handle standalone text input (no other modalities present)
        if has_text and not (has_imgs or has_pois or has_building or has_svis):
            for i, txt_type in enumerate(text_list):
                text_features = self.prompt_feature_extractor[i](text[txt_type])
                aligned_emb = self.cross_modality_alignment[i](text_features, text_features)
                features_list.append(aligned_emb)
                modal_features[txt_type] = aligned_emb
        
        # Return only the extracted and aligned features
        if only_feature:
            return modal_features