"""
LSTM Model for Time Series Classification and Early Failure Prediction.
"""

import torch
import torch.nn as nn
from typing import Optional


class Model(nn.Module):
    """
    LSTM-based model for time series classification.
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = getattr(configs, 'pred_len', 0)
        self.enc_in = configs.enc_in
        self.num_class = getattr(configs, 'num_class', 2)

        # Model hyperparameters
        self.hidden_size = getattr(configs, 'd_model', 128)
        self.num_layers = getattr(configs, 'e_layers', 2)
        self.dropout = getattr(configs, 'dropout', 0.1)
        self.bidirectional = True

        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=self.enc_in,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=self.bidirectional
        )

        # Output dimension after LSTM
        lstm_output_size = self.hidden_size * (2 if self.bidirectional else 1)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, self.num_class)
        )

        # For forecasting tasks
        self.forecast_head = nn.Linear(lstm_output_size, self.enc_in)

    def classification(self, x_enc: torch.Tensor, x_mark_enc: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Classification forward pass.

        Args:
            x_enc: Input tensor (batch_size, seq_len, enc_in)
            x_mark_enc: Time marks (unused)

        Returns:
            Class logits (batch_size, num_class)
        """
        # Normalize input
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x_enc)

        # Use last hidden state from both directions
        if self.bidirectional:
            # Concatenate last hidden states from both directions
            hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            hidden = h_n[-1]

        # Classification
        logits = self.classifier(hidden)

        return logits

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """Forecasting forward pass."""
        # Normalize
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        # LSTM forward
        lstm_out, _ = self.lstm(x_enc)

        # Forecast
        dec_out = self.forecast_head(lstm_out[:, -self.pred_len:, :])

        # Denormalize
        dec_out = dec_out * stdev + means

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """Forward pass with task routing."""
        if self.task_name == 'classification':
            return self.classification(x_enc, x_mark_enc)
        elif self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        else:
            return self.classification(x_enc, x_mark_enc)
