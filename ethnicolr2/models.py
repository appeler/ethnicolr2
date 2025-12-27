from __future__ import annotations

import torch
import torch.nn as nn


class LSTM(nn.Module):
    """LSTM model for ethnicity prediction from character sequences.

    Args:
        input_size: Size of vocabulary (number of unique characters)
        hidden_size: Size of hidden state in LSTM
        output_size: Number of output categories
        num_layers: Number of LSTM layers
    """

    def __init__(
        self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 1
    ) -> None:
        if input_size <= 0:
            raise ValueError(f"input_size must be positive, got {input_size}")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if output_size <= 0:
            raise ValueError(f"output_size must be positive, got {output_size}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")

        super().__init__()  # type: ignore[misc]
        self.hidden_size: int = hidden_size
        self.num_layers: int = num_layers

        self.embedding: nn.Embedding = nn.Embedding(input_size, hidden_size)
        self.lstm: nn.LSTM = nn.LSTM(
            hidden_size, hidden_size, num_layers, batch_first=True
        )
        self.fc: nn.Linear = nn.Linear(hidden_size, output_size)
        self.softmax: nn.LogSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass through the LSTM model.

        Args:
            input: Tensor of character indices with shape (batch_size, seq_len)

        Returns:
            Log-softmax probabilities for each category with shape (batch_size, output_size)
        """
        if not isinstance(input, torch.Tensor):  # type: ignore[arg-type]
            raise TypeError(f"Expected torch.Tensor, got {type(input)}")
        if len(input.shape) != 2:
            raise ValueError(
                f"Expected 2D tensor (batch_size, seq_len), got shape {input.shape}"
            )

        embedded = self.embedding(input.to(torch.int32))
        h0 = torch.zeros(self.num_layers, embedded.size(0), self.hidden_size).to(
            input.device
        )
        c0 = torch.zeros(self.num_layers, embedded.size(0), self.hidden_size).to(
            input.device
        )
        out, _ = self.lstm(embedded, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        out = self.softmax(out)
        return out
