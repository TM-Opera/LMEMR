import os
import torch
import numpy as np
from PIL import Image, ImageFile
from torchvision import models, transforms
import random

# Set GPU device
torch.cuda.set_device(3)

# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# -------------------------------
# Configuration parameters
# -------------------------------
SOURCE_FOLDER = r""  # Modify to your actual path
IMAGE_SIZE = 224
SEED = 32  # Base seed
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN_DIM = 2048  # Feature dimension of ResNet-152

# -------------------------------
# Image preprocessing
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet normalization
])

# -------------------------------
# Load pre-trained ResNet-152 model (with frozen parameters)
# -------------------------------
print("Loading pre-trained ResNet-152...")
model = models.resnet152(weights="IMAGENET1K_V1")  # Use pre-trained weights
model.fc = torch.nn.Identity()  # Remove the final fully connected layer to extract 2048-dimensional features
model = model.to(DEVICE)
model.eval()  # Set to evaluation mode (freeze)

print(f"ResNet-152 feature dim: {HIDDEN_DIM}")

# -------------------------------
# Extract single image feature (returns None on failure)
# -------------------------------
def extract_image_feature(img_path, model, transform, device):
    try:
        with Image.open(img_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model(img_tensor)
        return feat.cpu().numpy().squeeze(0)  # Return (2048,) vector
    except Exception as e:
        print(f"❌ Failed to process {img_path}: {e}")
        return None

# -------------------------------
# Main processing function: Remote sensing image feature extraction
# -------------------------------
def process_remote_sensing_images(source_folder):
    print(f"Processing remote sensing images in: {source_folder}")
    
    region_folders = [f for f in os.listdir(source_folder) if os.path.isdir(os.path.join(source_folder, f))]
    print(f"Found {len(region_folders)} region folders.")

    processed_count = 0
    failed_dirs = []

    for gwbh in region_folders:
        region_path = os.path.join(source_folder, gwbh)
        rs_image_name = f"{gwbh}_remote_sensing.png"
        rs_image_path = os.path.join(region_path, rs_image_name)

        # Initialize features: all zeros, shape (2048,)
        features = np.zeros(HIDDEN_DIM, dtype=np.float32)

        # Check if remote sensing image exists
        if not os.path.exists(rs_image_path):
            print(f"⚠️  Remote sensing image not found: {rs_image_path}")
            output_path = os.path.join(region_path, f"{gwbh}_remote_sensing_code.npy")
            np.save(output_path, features)
            failed_dirs.append(gwbh)
            continue

        # Extract features
        feat = extract_image_feature(rs_image_path, model, transform, DEVICE)
        if feat is not None:
            features = feat
            success_msg = "✅"
        else:
            success_msg = "⚠️ (Failed, saved zeros)"

        # Save features
        output_path = os.path.join(region_path, f"{gwbh}_remote_sensing_code.npy")
        np.save(output_path, features)
        print(f"{success_msg} Saved: {output_path}")
        processed_count += 1

    # Summary
    print(f"\n==================================")
    print(f"✅ Remote sensing feature extraction completed!")
    print(f"   Successfully processed: {processed_count} regions")
    print(f"   Failed/missing: {len(failed_dirs)} regions (saved as zero vectors)")
    print(f"   Total: {len(region_folders)} regions")
    print(f"==================================")

# -------------------------------
# Execute
# -------------------------------
if __name__ == "__main__":
    process_remote_sensing_images(SOURCE_FOLDER)