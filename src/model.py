import torch
import torch.nn as nn
import math

# POSICIONAL ENCODING

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1)]
        return x


# TRANSFORMER MODEL

class TransformerModel(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        pad_id=0
    ):
        super().__init__()

        self.d_model = d_model
        self.pad_id = pad_id

        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_id)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_id)

        # Positional Encoding
        self.positional_encoding = PositionalEncoding(d_model)

        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        # Output projection
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

        self.dropout = nn.Dropout(dropout)

    # MÁSCARAS

    def make_src_padding_mask(self, src):
        """
        src: [batch_size, src_len]
        """
        return src == self.pad_id

    def make_tgt_padding_mask(self, tgt):
        """
        tgt: [batch_size, tgt_len]
        """
        return tgt == self.pad_id

    def make_tgt_subsequent_mask(self, tgt_len):
        """
        Impede o decoder de ver tokens futuros
        """
        return torch.triu(
            torch.ones(tgt_len, tgt_len) * float("-inf"),
            diagonal=1
        )

    # FORWARD

    def forward(self, src, tgt):
        """
        src: [batch_size, src_len]
        tgt: [batch_size, tgt_len]
        """

        src_padding_mask = self.make_src_padding_mask(src)
        tgt_padding_mask = self.make_tgt_padding_mask(tgt)
        tgt_sub_mask = self.make_tgt_subsequent_mask(tgt.size(1)).to(tgt.device)

        # Embedding + escala
        src = self.src_embedding(src) * math.sqrt(self.d_model)
        tgt = self.tgt_embedding(tgt) * math.sqrt(self.d_model)

        # Positional Encoding
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        src = self.dropout(src)
        tgt = self.dropout(tgt)

        # Transformer
        output = self.transformer(
            src,
            tgt,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            tgt_mask=tgt_sub_mask
        )

        # Projeção final
        output = self.fc_out(output)

        return output
