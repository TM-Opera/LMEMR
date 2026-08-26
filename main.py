import torch
import argparse
from torchvision import transforms
from torch.utils.data import DataLoader

# Function/Model definitions
from model.Multimodal_Semantic_Enhancer import Multimodal_Semantic_Enhancer
from model.TrainWithGraph import train_modelwithGAT
from model.DataSet import Test_FlowDataset


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Multimodal Semantic Enhancer Model Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Basic training:                              python main.py --root_dir /path/to/data
  Custom training parameters:                 python main.py --epochs 400 --lr 0.001 --batch_size 32
  Use only specific modalities:                python main.py --no_text --no_svis
  CPU training:                               python main.py --device cpu
  Full configuration:                         python main.py --root_dir ./data --epochs 300 --lr 1e-4 --GAT_dim 256 --fusion_type attention
        """
    )

    # ========== Data Path Configuration ==========
    data_group = parser.add_argument_group('Data Config')
    data_group.add_argument('--root_dir', type=str, default='',
        help='Root directory of the dataset containing all required data files')
    data_group.add_argument('--num_bins', type=int, default=10,
        help='Number of bins for data discretization')

    # ========== Device Configuration ==========
    device_group = parser.add_argument_group('Device Config')
    device_group.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
        help='Training device: cuda (GPU acceleration) or cpu')

    # ========== Training Configuration ==========
    train_group = parser.add_argument_group('Training Config')
    train_group.add_argument('--epochs', type=int, default=400,
        help='Number of training epochs (total iterations)')
    train_group.add_argument('--lr', type=float, default=1e-4,
        help='Learning rate, controls parameter update step size')
    train_group.add_argument('--batch_size', type=int, default=None,
        help='Batch size, set to None to use full dataset as single batch')

    # ========== Model Configuration ==========
    model_group = parser.add_argument_group('Model Config')
    model_group.add_argument('--GAT_dim', type=int, default=256,
        help='Hidden dimension of Graph Attention Network (GAT)')
    model_group.add_argument('--use_GAT', type=str, default='resGAT',
        choices=['resGAT', 'GAT', 'none'],
        help='GAT type: resGAT (residual connection), GAT (standard), none (disabled)')
    model_group.add_argument('--num_modalities', type=int, default=4,
        help='Number of modalities for multimodal fusion (actual enabled modalities)')
    model_group.add_argument('--fusion_type', type=str, default='attention',
        choices=['attention', 'concat', 'weighted'],
        help='Multimodal fusion strategy: attention (attention mechanism), concat (concatenation), weighted (weighted sum)')
    model_group.add_argument('--classifier_arch', type=str, default='transformer',
        choices=['transformer', 'mlp'],
        help='Classifier architecture: transformer (self-attention based) or mlp (multi-layer perceptron)')

    # ========== Data Split Configuration ==========
    split_group = parser.add_argument_group('Data Split')
    split_group.add_argument('--train_ratio', type=float, default=0.6,
        help='Training set ratio (0.0-1.0), remaining split by val_ratio for validation and test')
    split_group.add_argument('--val_ratio', type=float, default=0.2,
        help='Validation set ratio (0.0-1.0), test set ratio = 1 - train_ratio - val_ratio')

    # ========== Loss Function Parameters ==========
    loss_group = parser.add_argument_group('Loss Config')
    loss_group.add_argument('--CL_gamma', type=float, default=0.1,
        help='Contrastive Learning loss weight, controls contrastive loss contribution in total loss')
    loss_group.add_argument('--temperature', type=float, default=0.1,
        help='Temperature parameter for softmax scaling, smaller values create steeper distributions')

    # ========== Modality Toggles ==========
    modal_group = parser.add_argument_group('Modality Toggles')
    modal_group.add_argument('--use_images', action='store_true', default=True,
        help='Enable image modality features (enabled by default)')
    modal_group.add_argument('--use_pois', action='store_true', default=True,
        help='Enable POI (Point of Interest) modality features (enabled by default)')
    modal_group.add_argument('--use_building', action='store_true', default=True,
        help='Enable building modality features (enabled by default)')
    modal_group.add_argument('--use_svis', action='store_true', default=True,
        help='Enable SVIS (Scene Visual) modality features (enabled by default)')
    modal_group.add_argument('--use_text', action='store_true', default=True,
        help='Enable text modality features (enabled by default)')

    # ========== Disable Modality Parameters ==========
    modal_group.add_argument('--no_images', action='store_false', dest='use_images',
        help='Disable image modality')
    modal_group.add_argument('--no_pois', action='store_false', dest='use_pois',
        help='Disable POI modality')
    modal_group.add_argument('--no_building', action='store_false', dest='use_building',
        help='Disable building modality')
    modal_group.add_argument('--no_svis', action='store_false', dest='use_svis',
        help='Disable SVIS modality')
    modal_group.add_argument('--no_text', action='store_false', dest='use_text',
        help='Disable text modality')

    # ========== Other Configuration ==========
    other_group = parser.add_argument_group('Other Config')
    other_group.add_argument('--visualize_embeddings', action='store_true',
        help='Enable embedding visualization, generates t-SNE/UMAP dimensionality reduction plots after training')
    other_group.add_argument('--use_amp', action='store_true', default=True,
        help='Enable Automatic Mixed Precision (AMP), accelerates training and reduces memory usage')
    other_group.add_argument('--no_amp', action='store_false', dest='use_amp',
        help='Disable automatic mixed precision training')

    # ========== Save Paths ==========
    path_group = parser.add_argument_group('Save Paths')
    path_group.add_argument('--save_path', type=str, default='model.pth',
        help='Path to save model weights after training (.pth format)')
    path_group.add_argument('--metrics_save_path', type=str, default='metrics.npz',
        help='Path to save training metrics (accuracy, F1, etc.) (.npz format)')
    path_group.add_argument('--confusion_matrix_path', type=str, default='matrix.npz',
        help='Path to save confusion matrix (.npz format)')

    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()

    # Data path configuration
    root_dir = args.root_dir
    num_bins = args.num_bins

    # Create dataset
    dataset = Test_FlowDataset(root_dir, fit_processor=True, num_bins=num_bins)
    Processor = dataset.get_processor()
    process_value = Processor.get_process_values()

    # Batch size configuration
    batch_size = args.batch_size if args.batch_size is not None else int(len(dataset))

    full_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # Configuration dictionary
    config = {
        'device': torch.device(args.device),
        'epochs': args.epochs,
        'lr': args.lr,
        'save_path': args.save_path,
        'bin_centers': process_value['bin'],
        'metrics_save_path': args.metrics_save_path,
        'confusion_matrix_path': args.confusion_matrix_path,
        'GAT_dim': args.GAT_dim,
        'use_GAT': args.use_GAT,
        'train_ratio': args.train_ratio,
        'val_ratio': args.val_ratio,
        'CL_gamma': args.CL_gamma,
        'temperature': args.temperature,
        'use_images': args.use_images,
        'use_pois': args.use_pois,
        'use_building': args.use_building,
        'use_svis': args.use_svis,
        'use_text': args.use_text,
        'num_modalities': args.num_modalities,
        'fusion_type': args.fusion_type,
        'visualize_embeddings': args.visualize_embeddings,
        'use_amp': args.use_amp,
        'classifier_arch': args.classifier_arch,
        'processor': Processor
    }

    # Initialize model
    model = Multimodal_Semantic_Enhancer(
        fusion_num=config['num_modalities'],
        feature_dim=256
    )

    # Train model
    trained_model = train_modelwithGAT(model, full_loader, config)


if __name__ == '__main__':
    main()
