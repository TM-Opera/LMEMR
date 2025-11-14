import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------ Loss Function Module ------------------

# Contrastive Loss
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature  # Temperature scaling for similarity

    def forward(self, img_feats, txt_feats):
        """
        Compute symmetric contrastive loss between image and text features.

        Args:
            img_feats: Tensor of shape [batch_size, C], image modality features
            txt_feats: Tensor of shape [batch_size, C], text modality features
        
        Returns:
            loss: scalar tensor representing the contrastive loss
        """
        batch_size = img_feats.shape[0]
 
        # L2 normalize features along the feature dimension
        img_feats = F.normalize(img_feats, dim=1)
        txt_feats = F.normalize(txt_feats, dim=1)

        # Compute cosine similarity between image and text features
        # sim[i][j] represents the similarity between i-th image and j-th text
        sim = torch.matmul(img_feats, txt_feats.t()) / self.temperature  # [B, B]

        # Create labels: diagonal elements are positive pairs (matching image-text)
        labels = torch.arange(batch_size).to(img_feats.device)

        # Symmetric loss: image-to-text retrieval + text-to-image retrieval
        loss_i2t = F.cross_entropy(sim, labels)  # Image to Text
        loss_t2i = F.cross_entropy(sim.t(), labels)  # Text to Image

        loss = (loss_i2t + loss_t2i) / 2
        return loss
    

# Multi-task Loss (Classification + Regression)
class ClassificationRegressionLoss(nn.Module):
    def __init__(self, alpha=0.5, device='cuda'):
        """
        Initialize multi-task loss with dynamic weighting.

        Args:
            alpha: initial weight balancing classification and regression losses
            device: device to place loss components on
        """
        super().__init__()
        self.classification_loss = nn.CrossEntropyLoss().to(device)
        self.regression_loss = nn.MSELoss().to(device)
        self.alpha = alpha
        # Learnable weights to replace fixed alpha for adaptive loss balancing
        self.learnable_weights = nn.Parameter(torch.tensor([alpha, 1-alpha]),
                                            requires_grad=True).to(device)
        
    def forward(self, class_logits, reg_preds, class_labels, reg_targets):
        """
        Forward pass to compute combined classification and regression loss.

        Args:
            class_logits: output logits from classification head [batch_size, num_classes]
            reg_preds: predicted regression values [batch_size]
            class_labels: ground truth class labels [batch_size]
            reg_targets: ground truth regression targets [batch_size]

        Returns:
            total_loss: scalar tensor combining both losses with learnable weights
        """
        # Classification loss
        class_loss = self.classification_loss(class_logits, class_labels)
        
        # Regression loss
        reg_loss = self.regression_loss(reg_preds, reg_targets)

        # Dynamic weight balancing using softmax normalization
        weights = torch.softmax(self.learnable_weights, dim=0)
        
        # Combined total loss
        total_loss = weights[0] * class_loss + weights[1] * reg_loss

        return total_loss