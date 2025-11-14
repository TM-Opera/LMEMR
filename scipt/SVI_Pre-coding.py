import os
import torch
import numpy as np
from PIL import Image, ImageFile
from torchvision import models, transforms

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# -------------------------------
# Configuration parameters
# -------------------------------
SOURCE_FOLDER = r""  # Modify to your actual path
IMAGE_SIZE = 224
NUM_IMAGES_PER_REGION = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN_DIM = 2048  # Hidden dimension of ResNet-152

# Fix CUDA device
torch.cuda.set_device(3)
DEVICE = torch.device("cuda:3")

# Set seed for reproducibility
SEED = 32
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

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
model = models.resnet152(weights="IMAGENET1K_V1")  # Use pre-trained ImageNet weights
model.fc = torch.nn.Identity()  # Remove final classification layer to extract 2048-dim features
model = model.to(DEVICE)
model.eval()  # Set to evaluation mode (frozen)

print(f"ResNet-152 feature dim: {HIDDEN_DIM}")

# -------------------------------
# Extract feature from a single image (returns None on failure)
# -------------------------------
def extract_image_feature(img_path, model, transform, device):
    try:
        with Image.open(img_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)  # Add batch dimension and move to device
        with torch.no_grad():
            feat = model(img_tensor)
        return feat.cpu().numpy().squeeze(0)  # Return feature vector of shape (2048,)
    except Exception as e:
        print(f"⚠️ Failed to process image: {img_path} | Error: {e}")
        return None  # Return None if loading/decoding fails

# -------------------------------
# Main processing function: Extract street view image features
# -------------------------------
def process_all_regions(source_folder, num_images=20):
    print(f"Processing regions in: {source_folder}")
    
    region_folders = [f for f in os.listdir(source_folder) if os.path.isdir(os.path.join(source_folder, f))]
    print(f"Found {len(region_folders)} region folders.")

    processed_count = 0
    failed_dirs = []

    for gwbh in sorted(region_folders):  # Sort to ensure consistent processing order
        region_path = os.path.join(source_folder, gwbh)
        svi_folder_name = f"{gwbh}_streetviews"
        svi_folder = os.path.join(region_path, svi_folder_name)

        # Initialize features as zero array of shape (20, 2048)
        features = np.zeros((num_images, HIDDEN_DIM), dtype=np.float32)

        # Check if street view folder exists
        if not os.path.exists(svi_folder) or not os.path.isdir(svi_folder):
            print(f"⚠️ No streetview folder: {gwbh}")
            output_path = os.path.join(region_path, f"{gwbh}_svi_code.npy")
            np.save(output_path, features)
            failed_dirs.append(gwbh)
            continue

        # Get all jpg/jpeg files and sort them alphabetically (for reproducibility)
        img_files = [
            f for f in os.listdir(svi_folder)
            if f.lower().endswith(('.jpg', '.jpeg')) and os.path.isfile(os.path.join(svi_folder, f))
        ]
        img_files = sorted(img_files)  # Ensure consistent order

        if len(img_files) == 0:
            print(f"⚠️ No images in: {gwbh}")
            output_path = os.path.join(region_path, f"{gwbh}_svi_code.npy")
            np.save(output_path, features)
            failed_dirs.append(gwbh)
            continue

        # Select up to the first 20 images
        selected_files = img_files[:num_images]

        # Extract features; keep zeros for failed ones
        success_count = 0
        for filename in selected_files:
            img_path = os.path.join(svi_folder, filename)
            feat = extract_image_feature(img_path, model, transform, DEVICE)
            if feat is not None:
                features[success_count] = feat
                success_count += 1
            # Otherwise, remains zero (already initialized)

        # Save extracted features
        output_path = os.path.join(region_path, f"{gwbh}_svi_code.npy")
        np.save(output_path, features)
        print(f"✅ Saved: {output_path} -> Shape: {features.shape} ({success_count}/{len(selected_files)} valid)")
        processed_count += 1

    # Summary
    print(f"\n==================================")
    print(f"✅ All regions processed!")
    print(f"   Successfully extracted features for: {processed_count} regions")
    print(f"   Missing/skipped: {len(failed_dirs)} regions (saved as zero-filled arrays)")
    print(f"   Total regions: {len(region_folders)}")
    print(f"==================================")

# -------------------------------
# Execute main function
# -------------------------------
if __name__ == "__main__":
    process_all_regions(SOURCE_FOLDER, num_images=NUM_IMAGES_PER_REGION)