import numpy as np

# ------------------ Flow Data Processing ------------------

class FlowDataProcessor:
    """Flow data processor: implements log transformation, Z-Score normalization, and discretization"""
    
    def __init__(self, num_bins=10, epsilon=1e-8):
        """
        Initialize the processor
        
        Parameters:
            num_bins: number of bins for discretization
            epsilon: small value to prevent division by zero
        """
        self.num_bins = num_bins
        self.epsilon = epsilon
        self.mean = None
        self.std = None
        self.bin_edges = None
        self.bin_centers = None
        self.normalized_data = []
    
    def fit(self, flow_data):
        """
        Fit distribution parameters from data
        
        Parameters:
            flow_data: raw flow data, shape [number_of_samples, 1]
        """
        self.normalized_data = []
        # Ensure data is a 2D array
        flow_data = np.array(flow_data).reshape(-1, 1)

        raw_values = flow_data.flatten()  # Extract original 1D array
            
        # Log transformation: log(x + 1) for non-negative data
        log_transformed = np.log(flow_data + 1)
        
        # Compute statistics
        self.mean = np.mean(log_transformed, axis=0)
        self.std = np.std(log_transformed, axis=0) + self.epsilon  # Prevent division by zero
        
        # Z-Score normalization
        normalized = (log_transformed - self.mean) / self.std
        self.normalized_data.extend(normalized.flatten().tolist())
        
        # Compute bin edges (using global range)
        min_val, max_val = np.min(normalized), np.max(normalized)
        
        self.bin_edges = np.linspace(min_val, max_val, self.num_bins + 1)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
        
        return self
    
    def transform(self, flow_data):
        """
        Apply transformations to the data
        
        Parameters:
            flow_data: raw flow data, shape [number_of_samples, 1]
            
        Returns:
            dict: transformed data in various forms
        """
        # Ensure data is a 2D array
        flow_data = np.array(flow_data).reshape(-1, 1)
            
        # Log transformation
        log_transformed = np.log(flow_data + 1)
        
        # Z-Score normalization
        normalized = (log_transformed - self.mean) / self.std
        
        # Discretization (classification)
        labels = np.digitize(normalized, self.bin_edges) - 1
        # Handle boundary values
        labels[labels < 0] = 0
        labels[labels >= self.num_bins] = self.num_bins - 1
        
        return {
            'log_transformed': log_transformed,
            'normalized': normalized,
            'labels': labels,
            'bin_centers': self.bin_centers
        }
    
    def inverse_transform(self, normalized_data, type='log'):
        """
        Inverse transform normalized data back to original or log-transformed scale

        Parameters:
            normalized_data: normalized data
            type: if 'log', return log-transformed data; otherwise return original scale
            
        Returns:
            np.ndarray: original or unnormalized data
        """
        # Reverse standardization
        log_transformed = normalized_data * self.std + self.mean
        
        # Reverse log transformation
        original = np.exp(log_transformed) - 1
        
        return log_transformed if type == 'log' else original

    def get_process_values(self):
        """
        Get key processing parameters
        
        Returns:
            dict: contains bin centers, mean, and std used in transformation
        """
        values = {
            'bin': self.bin_centers,
            'mean': self.mean,
            'std': self.std
        }
        return values

    def classify_to_regress(self, class_probs):
        """
        Convert classification probabilities to regression values (classification-to-regression strategy)
        
        Parameters:
            class_probs: probability distribution over classes, shape [number_of_samples, num_classes]
            
        Returns:
            np.ndarray: regression prediction values
        """
        # Weighted average: probability * bin center values
        return np.sum(class_probs * self.bin_centers, axis=-1)  

    def print_bin_ranges(self, prefix="Bin"):
        """
        Print the numerical range of each bin
        
        Parameters:
            prefix: name prefix for each bin
        """
        for i in range(len(self.bin_edges) - 1):
            lower = self.inverse_transform(self.bin_edges[i]).item()
            upper = self.inverse_transform(self.bin_edges[i + 1]).item()
            print(f"{prefix} {i}: [{lower:.0f}, {upper:.0f})")

    def count_samples_in_bins(self):
        """
        Count how many "time steps" fall into each bin
        
        Returns:
            bin_counts: array of shape [num_bins,], number of samples in each bin
        """
        normalized_data = np.array(self.normalized_data)
        # Ensure it's a 2D array
        if normalized_data.ndim == 1:
            normalized_data = normalized_data.reshape(-1, 1)

        # Use digitize to find which bin each sample falls into
        labels = np.digitize(normalized_data, self.bin_edges) - 1

        # Handle boundary cases (values outside bin range)
        labels[labels < 0] = 0
        labels[labels >= self.num_bins] = self.num_bins - 1

        # Count number of samples in each bin
        bin_counts = np.bincount(labels.flatten(), minlength=self.num_bins)

        # Print results
        for i, count in enumerate(bin_counts):
            print(f"Bin {i}: {count} samples")

        return bin_counts