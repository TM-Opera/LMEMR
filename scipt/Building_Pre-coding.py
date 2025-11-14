import os
import torch
import numpy as np
from PIL import Image, ImageFile
import timm
from torchvision import transforms
import random

# Set GPU device
torch.cuda.set_device(3)

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# -------------------------------
# Configuration parameters
# -------------------------------
SOURCE_FOLDER = r""  # Modify to your actual data path
IMAGE_SIZE = 224
SEED = 32  # Base seed for reproducibility
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN_DIM = 2048  # Feature dimension of ResNet-152

# Set global seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# -------------------------------
# Image preprocessing (for single-channel grayscale images)
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize single channel to [-1, 1]
])

# -------------------------------
# Load pre-trained ResNet-152 with single-channel input
# -------------------------------
print("Loading pre-trained ResNet-152 (single-channel) from timm...")
model = timm.create_model(
    "resnet152",
    pretrained=True,      # Use ImageNet pre-trained weights
    in_chans=1,           # Key: change input to single channel (grayscale)
    num_classes=1000      # Not used; will be removed
)

# Remove classification head to extract 2048-dimensional features
model.fc = torch.nn.Identity()
model = model.to(DEVICE)
model.eval()

# Freeze all parameters (feature extraction mode)
for param in model.parameters():
    param.requires_grad = False

print(f"ResNet-152 (single-channel) feature dim: {HIDDEN_DIM}")

# -------------------------------
# Extract feature from a single image (returns None on failure)
# -------------------------------
def extract_image_feature(img_path, model, transform, device):
    try:
        with Image.open(img_path) as img:
            if img.mode != 'L':
                img = img.convert('L')  # Convert to grayscale
            img_tensor = transform(img).unsqueeze(0).to(device)  # Shape: (1, 1, 224, 224)
        with torch.no_grad():
            feat = model(img_tensor)
        return feat.cpu().numpy().squeeze(0)  # Return 2048-dim vector
    except Exception as e:
        print(f"❌ Failed to process {img_path}: {e}")
        return None

# -------------------------------
# Main function: Building height map feature extraction
# -------------------------------
def process_building_images(source_folder):
    print(f"Processing building height images in: {source_folder}")
    
    region_folders = [f for f in os.listdir(source_folder) if os.path.isdir(os.path.join(source_folder, f))]
    print(f"Found {len(region_folders)} region folders.")

    processed_count = 0
    failed_dirs = []

    for gwbh in region_folders:
        region_path = os.path.join(source_folder, gwbh)
        img_name = f"{gwbh}_building.png"
        img_path = os.path.join(region_path, img_name)

        # Initialize features as zero vector of shape (2048,)
        features = np.zeros(HIDDEN_DIM, dtype=np.float32)

        # Check if the building image exists
        if not os.path.exists(img_path):
            print(f"⚠️  Building image not found: {img_path}")
            output_path = os.path.join(region_path, f"{gwbh}_building_code.npy")
            np.save(output_path, features)
            failed_dirs.append(gwbh)
            continue

        # Extract features
        feat = extract_image_feature(img_path, model, transform, DEVICE)
        if feat is not None:
            features = feat
            msg = "✅"
        else:
            msg = "⚠️ (Failed, saved zeros)"

        # Save extracted features
        output_path = os.path.join(region_path, f"{gwbh}_building_code.npy")
        np.save(output_path, features)
        print(f"{msg} Saved: {output_path}")
        processed_count += 1

    # Summary report
    print(f"\n==================================")
    print(f"✅ Building height map feature extraction completed!")
    print(f"   Successfully processed: {processed_count} regions")
    print(f"   Missing/failed: {len(failed_dirs)} regions (saved as zero vectors)")
    print(f"   Total regions: {len(region_folders)}")
    print(f"==================================")

# -------------------------------
# Execute main function
# -------------------------------
if __name__ == "__main__":
    process_building_images(SOURCE_FOLDER)