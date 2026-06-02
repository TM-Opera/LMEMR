import torch
from torch.utils.data import DataLoader

# Import function/models
from Multimodal_Semantic_Enhancer import Multimodal_Semantic_Enhancer
from TrainWithGraph import train_modelwithGAT
from DataSet import FlowDataset

# Data path
root_dir = ''   # Set your data root directory here
num_bins = 10

# Create dataset
dataset = FlowDataset(root_dir, fit_processor=True, num_bins=num_bins) 
processor = dataset.get_processor()
process_value = processor.get_process_values()
bin_centers = process_value['bin']

# Create full-batch data loader (batch size equals total dataset size)
full_loader = DataLoader(
    dataset,
    batch_size=int(len(dataset)),
    shuffle=False  # No shuffling for full-batch training with fixed graph
)

# LMEMR Configuration
config = {
    'device': torch.device('cuda'),                     # Device for training
    'epochs': 400,                                      # Total number of training epochs
    'lr': 1e-4,                                         # Learning rate
    'save_path': 'LMEMR.pth',                           # Path to save the best model
    'bin_centers': process_value['bin'],                # Bin centers for regression-to-classification conversion
    'metrics_save_path': 'Results/LMEMR_metrics.npz',   # Path to save training metrics
    'confusion_matrix_path': 'confusion_matrix/LMEMR_matrix.npz',  # Path to save confusion matrices
    'GAT_dim': 256,                                     # Feature dimension for GNN
    'use_GAT': 'resGAT',                                # Type of GNN: 'resGAT', 'GAT', 'GCN', or None
    'train_ratio': 0.6,                                 # Proportion of training set
    'val_ratio': 0.2,                                   # Proportion of validation set
    'CL_gamma': 0.1,                                    # Weight for contrastive loss
    'temperature': 0.1,                                 # Temperature parameter in contrastive loss
    'use_images': True,                                 # Whether to use remote sensing images
    'use_pois': True,                                   # Whether to use POI features
    'use_building': True,                               # Whether to use building height images
    'use_svis': True,                                   # Whether to use street view images
    'use_text': True,                                   # Whether to use text prompts
    'num_modalities': 4,                                # Number of modalities used
    'fusion_type': 'attention',                         # Fusion strategy: 'attention', 'concat', etc.
    'visualize_embeddings': False,                      # Whether to visualize embeddings during training
    'use_amp': True,                                    # Whether to use automatic mixed precision
    'classifier_arch': 'transformer',                   # Architecture of the classification head
    'processor': processor                              # Data processor for inverse transformation
}

# Initialize the multimodal semantic enhancer
model = Multimodal_Semantic_Enhancer(fusion_num=config['num_modalities'], feature_dim=256)

# Train and evaluate the model
trained_model = train_modelwithGAT(model, full_loader, config)
