import torch
import torch.nn as nn
from Classifier import MLPClassifier, LSTMClassifier, CNNClassifier, TransformerClassifier

# ------------------ Custom Classifier Wrapper ------------------
#
# A unified classifier that selects and wraps different architectures
# based on the provided 'arch' argument.
# Input: [B, input_dim]  --> Output: [B, 24, output_dim]


class SelfClassifier(nn.Module):
    """
    A flexible wrapper that instantiates and uses different classifier architectures
    such as MLP, LSTM, CNN, or Transformer for multi-time-step classification.

    The input is first expanded across a time dimension using a 1D convolution,
    then passed to the selected model.
    """
    def __init__(
        self,
        arch='transformer',  # Supported: 'mlp', 'lstm', 'cnn', 'transformer'
        input_dim=256,
        output_dim=10,
        sequence_length=24,
        # Architecture-specific parameters
        mlp_hidden_dim=256,
        lstm_hidden_dim=256,
        gru_hidden_dim=256,  # Note: GRU not used in current implementation
        cnn_num_channels=256,
        transformer_d_model=256,
        nhead=4,
        num_layers=12,
        dim_feedforward=1024,
        dropout=0.2
    ):
        super().__init__()
        self.arch = arch.lower()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sequence_length = sequence_length

        # 1D convolution to expand scalar input to a sequence of length 24
        # Input: [B, 1, input_dim] → After conv: [B, 24, input_dim]
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=sequence_length, kernel_size=1)

        # Initialize the selected architecture
        if self.arch == 'mlp':
            self.model = MLPClassifier(
                input_dim=input_dim,
                hidden_dim=mlp_hidden_dim,
                output_dim=output_dim,
                num_layers=1,
                dropout=dropout
            )
        elif self.arch == 'lstm':
            self.model = LSTMClassifier(
                input_dim=input_dim,
                hidden_dim=lstm_hidden_dim,
                output_dim=output_dim,
                num_layers=2,
                dropout=dropout
            )
        elif self.arch == 'cnn':
            self.model = CNNClassifier(
                input_dim=input_dim,
                num_channels=cnn_num_channels,
                output_dim=output_dim,
                num_blocks=1,
                dropout=dropout
            )
        elif self.arch == 'transformer':
            self.model = TransformerClassifier(
                feature_dim=input_dim,
                num_bins=output_dim,
                d_model=transformer_d_model,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unsupported architecture: {arch}")

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape [B, input_dim]

        Returns:
            torch.Tensor: Output logits of shape [B, 24, output_dim]
        """
        # Expand input to sequence via 1D convolution
        x = x.unsqueeze(1)  # [B, 1, input_dim]
        x = self.conv1(x)   # [B, 24, input_dim]

        # Pass through selected model
        return self.model(x)


# Example usage and test
if __name__ == "__main__":
    # Test with dummy data
    batch_size = 8
    input_dim = 256
    output_dim = 10
    seq_len = 24

    x = torch.randn(batch_size, input_dim)  # Simulated input

    # Try each architecture
    for arch in ['mlp', 'lstm', 'cnn', 'transformer']:
        print(f"\nTesting {arch.upper()} model...")
        model = SelfClassifier(
            arch=arch,
            input_dim=input_dim,
            output_dim=output_dim
        )
        model.eval()
        with torch.no_grad():
            out = model(x)
        print(f"Output shape: {out.shape} ✅")  # Should be [8, 24, 10]