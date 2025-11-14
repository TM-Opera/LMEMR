import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from flowdataprocessor import FlowDataProcessor

# ------------------ Dataset Definition ------------------
#
# A PyTorch Dataset for loading multi-modal urban data per GWBH region.
# Supports remote sensing images, building height maps, POI features, street views,
# textual descriptions, and hourly human flow time series.

class FlowDataset(Dataset):
    """
    Dataset class that loads data from individual GWBH folders.
    Each folder contains multimodal inputs and a 24-hour flow time series.

    Supports optional loading of:
        - Remote sensing images
        - Building height images
        - Text embeddings (LLM-generated)
        - POI distributions
        - Street view images (SVI)
        - Pre-extracted image features (.npy)

    Args:
        root_dir (str): Root directory containing GWBH subfolders.
        load_remote (bool): Whether to load remote sensing images.
        load_building (bool): Whether to load building height images.
        flow_column (str): Name of the flow column (for compatibility; currently uses .npy).
        num_bins (int): Number of bins for flow quantization.
        rs_transform (callable): Transform for remote sensing images.
        building_transform (callable): Transform for building images.
        fit_processor (bool): Whether to fit the FlowDataProcessor on this dataset.
        shuffle_text (bool): Whether to randomly shuffle text feature order (for ablation).
    """

    def __init__(self, root_dir, load_remote=False, load_building=False,
                 flow_column='hourly_flow', num_bins=10, rs_transform=None,
                 building_transform=None, fit_processor=True, shuffle_text=False):
        self.root_dir = root_dir
        self.load_remote = load_remote
        self.load_building = load_building
        self.rs_transform = rs_transform
        self.building_transform = building_transform
        self.flow_column = flow_column
        self.processor = FlowDataProcessor(num_bins=num_bins)
        self.bin_centers = []
        self.shuffle_text = shuffle_text

        # Discover valid GWBH folders with required files
        self.gwbh_folders = self._get_valid_gwbh_folders()
        print(f"Found {len(self.gwbh_folders)} valid GWBH folders")

        # Fit flow statistics if requested
        if fit_processor:
            self._fit_flow_processor()

        if shuffle_text:
            print("⚠️ Note: Text feature shuffling is enabled.")

    def _get_valid_gwbh_folders(self):
        """Scan root directory and return list of GWBH folders with required data."""
        valid_folders = []
        for folder in os.listdir(self.root_dir):
            folder_path = os.path.join(self.root_dir, folder)
            if not os.path.isdir(folder_path):
                continue

            remote_img = os.path.join(folder_path, f"{folder}_remote_sensing.png")
            building_img = os.path.join(folder_path, f"{folder}_building.png")
            flow_file = os.path.join(folder_path, f"{folder}_flow.npy")

            has_remote = not self.load_remote or os.path.exists(remote_img)
            has_building = not self.load_building or os.path.exists(building_img)

            if has_remote and has_building and os.path.exists(flow_file):
                valid_folders.append(folder)
        return valid_folders

    def _fit_flow_processor(self):
        """Fit the FlowDataProcessor using all available flow sequences."""
        flow_data_list = []
        for gwbh in self.gwbh_folders:
            flow_file = os.path.join(self.root_dir, gwbh, f"{gwbh}_flow.npy")
            flow = np.load(flow_file)  # [24]
            flow_data_list.extend(flow)

        flow_array = np.array(flow_data_list)
        self.processor.fit(flow_array)
        self.bin_centers = self.processor.get_process_values()['bin']
        print(f"Flow processor fitted with {len(flow_array)} samples.")

    def get_bin_centers(self):
        """Return bin centers used for label dequantization."""
        return self.bin_centers

    def get_processor(self):
        """Return the fitted FlowDataProcessor instance."""
        return self.processor

    def __len__(self):
        return len(self.gwbh_folders)

    def _load_text_code(self, folder_path, gwbh, text_key):
        """Load pre-extracted LLM text embedding (e.g., LLM_RS_*.npy)."""
        text_path = os.path.join(folder_path, f"LLM_{text_key}_{gwbh}_code.npy")
        if not os.path.exists(text_path):
            return torch.zeros(4096, dtype=torch.float32)
        code = np.load(text_path)
        return torch.tensor(code, dtype=torch.float32).squeeze()  # [D]

    def _load_image(self, folder_path, gwbh, img_type):
        """Load and transform an image (remote sensing or building)."""
        img_path = os.path.join(folder_path, f"{gwbh}_{img_type}.png")
        if not os.path.exists(img_path):
            print(f'⚠️ Missing image: {gwbh}_{img_type}.png')

        try:
            if img_type == 'building':
                image = Image.open(img_path).convert('L')
                if self.building_transform:
                    image = self.building_transform(image)
            else:
                image = Image.open(img_path).convert('RGB')
                if self.rs_transform:
                    image = self.rs_transform(image)
            return image
        except Exception as e:
            print(f"❌ Failed to load image {img_path}: {e}")
            # Return zero tensor with expected shape
            c = 1 if img_type == 'building' else 3
            return torch.zeros(c, 224, 224, dtype=torch.float32)

    def _load_rs_code(self, folder_path, gwbh):
        """Load pre-extracted ResNet-152 feature for remote sensing image."""
        rs_path = os.path.join(folder_path, f"{gwbh}_remote_sensing_code.npy")
        if not os.path.exists(rs_path):
            return "None rs_code"
        code = np.load(rs_path)
        return torch.tensor(code, dtype=torch.float32).squeeze()  # [2048]

    def _load_building_code(self, folder_path, gwbh):
        """Load pre-extracted ResNet-152 feature for building height image."""
        bld_path = os.path.join(folder_path, f"{gwbh}_building_code.npy")
        if not os.path.exists(bld_path):
            return "None building_code"
        code = np.load(bld_path)
        return torch.tensor(code, dtype=torch.float32).squeeze()  # [2048]

    def _load_poi_data(self, folder_path, gwbh):
        """Load POI category distribution from CSV."""
        poi_file = os.path.join(folder_path, f"{gwbh}_poi_desc.csv")
        if not os.path.exists(poi_file):
            return torch.zeros((23), dtype=torch.float32)
        try:
            df = pd.read_csv(poi_file)
            values = df.iloc[:, 1:].values.flatten()
            return torch.tensor(values, dtype=torch.float32)
        except Exception as e:
            print(f"❌ Failed to load POI data for {gwbh}: {e}")
            return torch.zeros((23), dtype=torch.float32)

    def _load_svi_code(self, folder_path, gwbh):
        """Load pre-extracted street view image features (20 views)."""
        svi_path = os.path.join(folder_path, f"{gwbh}_svi_code.npy")
        if not os.path.exists(svi_path):
            return "None svi"
        code = np.load(svi_path)  # [20, D]
        return torch.tensor(code, dtype=torch.float32).squeeze()  # [20, D]

    def __getitem__(self, idx):
        """Return a single sample with all modalities."""
        gwbh = self.gwbh_folders[idx]
        folder_path = os.path.join(self.root_dir, gwbh)

        # 1. Load image data
        images = {}
        if self.load_remote:
            remote_img = self._load_image(folder_path, gwbh, 'remote_sensing')
            images['remote'] = remote_img
        if self.load_building:
            building_img = self._load_image(folder_path, gwbh, 'building')
            images['building'] = building_img

        # Load pre-extracted deep features
        imgs_code = {
            'rsi_code': self._load_rs_code(folder_path, gwbh),
            'building_code': self._load_building_code(folder_path, gwbh)
        }

        # Load auxiliary data
        poi_data = self._load_poi_data(folder_path, gwbh)
        svi_data = self._load_svi_code(folder_path, gwbh)

        # 2. Load raw flow
        flow_file = os.path.join(folder_path, f"{gwbh}_flow.npy")
        raw_flow = np.load(flow_file)  # [24]

        # 3. Normalize and bin flow
        processed = self.processor.transform(raw_flow)
        labels = torch.tensor(processed['labels'], dtype=torch.long)  # [24]
        normalized = torch.tensor(processed['normalized'].squeeze(-1), dtype=torch.float32)  # [24]

        raw_flow = torch.tensor(raw_flow, dtype=torch.float32)

        # 4. Load text embeddings
        text = {
            'rsi': self._load_text_code(folder_path, gwbh, 'RS'),
            'building': self._load_text_code(folder_path, gwbh, 'Building'),
            'poi': self._load_text_code(folder_path, gwbh, 'Pois'),
            'svi': self._load_text_code(folder_path, gwbh, 'SVI')
        }

        # Optional: Shuffle text feature order across batch dimension
        if self.shuffle_text:
            first_key = list(text.keys())[0]
            B = text[first_key].shape[0] if text[first_key].dim() > 0 else 1
            indices = torch.randperm(B)
            shuffled_text = {}
            for key, value in text.items():
                if isinstance(value, torch.Tensor) and value.size(0) == B:
                    shuffled_text[key] = value[indices]
                else:
                    print(f"⚠️ Cannot shuffle text feature '{key}', keeping original.")
                    shuffled_text[key] = value
            text = shuffled_text

        # 5. Construct final sample
        sample = {
            'imgs': images,                   # Transformed PIL images
            'img_code': imgs_code,           # Pre-extracted CNN features
            'pois': poi_data,                # POI distribution [23]
            'targets': {
                'original_flow': raw_flow,   # Raw hourly flow [24]
                'normalized': normalized,    # Normalized flow [24]
                'labels': labels             # Quantized labels [24]
            },
            'text': text,                    # LLM embeddings [4096] each
            'svi': svi_data,                 # Street view features [20, D]
            'gwbh': int(gwbh)                # Region ID
        }

        return sample