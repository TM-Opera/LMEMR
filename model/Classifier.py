import torch
import torch.nn as nn   
import torch.nn.functional as F 
import math

# #----------------- Classifier Models -----------------
#
# Input: [B, 24, 256]
# Output: [B, 24, 10]  (10-class classification for each of the 24 time steps)


class MLPClassifier(nn.Module):
    """
    A simple multi-layer perceptron (MLP) classifier with layer normalization and residual blocks.
    Each time step uses an independent output head.
    """
    def __init__(self, input_dim=256, hidden_dim=256, output_dim=10, num_layers=1, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Deep feed-forward blocks with LayerNorm and Dropout
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ) for _ in range(num_layers)
        ])
        
        # Independent output heads for each of the 24 time steps
        self.output_heads = nn.ModuleList([nn.Linear(hidden_dim, output_dim) for _ in range(24)])

    def forward(self, x):
        # x: [B, 24, 256]
        x = self.input_proj(x)  # [B, 24, hidden_dim]

        # Apply each block sequentially
        for block in self.blocks:
            x = x + block(x)  # Residual connection

        # Use separate head for each time step
        outputs = []
        for t in range(24):
            out_t = self.output_heads[t](x[:, t, :])  # [B, 10]
            outputs.append(out_t.unsqueeze(1))       # [B, 1, 10]

        # Concatenate along time dimension
        out = torch.cat(outputs, dim=1)  # [B, 24, 10]
        return out
    

class LSTMClassifier(nn.Module):
    """
    LSTM-based classifier that models temporal dependencies in the sequence.
    Uses LayerNorm and dropout for regularization.
    """
    def __init__(self, input_dim=256, hidden_dim=256, output_dim=10, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        self.pre_norm = nn.LayerNorm(input_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: [B, 24, 256]
        x = self.pre_norm(x)
        lstm_out, _ = self.lstm(x)                    # [B, 24, hidden_dim]
        out = self.norm(lstm_out)
        out = self.dropout(out)
        out = self.fc(out)                            # [B, 24, 10]
        return out


class ConvBlock(nn.Module):
    """
    1D Convolutional block with batch norm, ReLU, and dropout.
    No residual connection is used in this version.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                               padding=padding, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, C, T]
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.dropout(out)
        out = self.relu(out)
        return out  # No residual connection


class CNNClassifier(nn.Module):
    """
    1D CNN classifier using multiple convolutional blocks to capture local patterns.
    Processes sequence in temporal dimension via 1D convolutions.
    """
    def __init__(self, input_dim=256, num_channels=256, output_dim=10, num_blocks=1, dropout=0.2):
        super().__init__()
        self.input_conv = nn.Conv1d(input_dim, num_channels, 1)  # Point-wise projection

        self.blocks = nn.ModuleList([
            ConvBlock(num_channels, num_channels, kernel_size=3, dilation=1, dropout=dropout)
            for _ in range(num_blocks)
        ])

        self.global_norm = nn.LayerNorm(num_channels)
        self.output_proj = nn.Linear(num_channels, output_dim)

    def forward(self, x):
        # x: [B, 24, 256]
        x = x.transpose(1, 2)  # -> [B, 256, 24]

        x = self.input_conv(x)  # [B, C, T]

        for block in self.blocks:
            x = block(x)  

        x = x.transpose(1, 2)  # -> [B, 24, C]
        x = self.global_norm(x)
        out = self.output_proj(x)  # [B, 24, 10]
        return out


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for Transformer models.
    Adds position information to input embeddings.
    """
    def __init__(self, d_model, max_len=24):
        super().__init__()
        self.dropout = nn.Dropout(p=0.1)

        # Create [max_len, d_model] positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)
    

class TransformerClassifier(nn.Module):
    """
    Transformer-based classifier that models long-range dependencies in the time series.
    Uses multi-head self-attention and position encoding.
    """
    def __init__(self, feature_dim, num_bins=10, d_model=256, nhead=4, num_layers=12, dim_feedforward=1024, dropout=0.2):
        super().__init__()
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.dropout = dropout

        # Project input features to model dimension
        self.proj = nn.Linear(feature_dim, d_model)

        # Add positional encoding
        self.positional = PositionalEncoding(d_model)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=self.dropout,
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final classification head
        self.classifier = nn.Linear(d_model, num_bins)

    def forward(self, x):
        # x shape: (batch_size, 24, feature_dim) —— assumed corrected from docstring
        batch_size = x.size(0)

        # Project to model dimension
        x = self.proj(x)  # [B, 24, d_model]

        # Add positional encoding
        x = self.positional(x)

        # Pass through Transformer encoder
        x = self.encoder(x)  # [B, 24, d_model]

        # Final classification
        output_class = self.classifier(x)  # [B, 24, num_bins]

        return output_class