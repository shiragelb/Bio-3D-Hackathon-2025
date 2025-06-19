import torch
import torch.nn as nn
import math

# ---------------------------------------------------------------------
# NEW: helper that normal-initialises every Linear / Embedding weight
# ---------------------------------------------------------------------
def _init_weights(module, mean: float = 0.0, std: float = 0.02):
    """
    Apply N(0, std²) to all learnable parameters that benefit from it.
    • Linear, Embedding → weight ~ 𝒩(mean, std²)
    • bias → 0
    • LayerNorm → weight = 1, bias = 0   (standard LN init)
    """
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.normal_(module.weight, mean=mean, std=std)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard (fixed) sinusoidal positional encodings from
    No gradients/learned params.
    """

    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2) *
            -(math.log(10000.0) / d_model)
        )  # [d_model/2]

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len, d_model]
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class PeriodicModuloEncoding(nn.Module):
    """
    Learned absolute positions  +  learned 'mod-p' embeddings.
    Using known NES periods of (2-4).
    """

    def __init__(self, d_model: int, max_len: int, periods=(2, 3, 4)):
        super().__init__()
        self.abs = nn.Embedding(max_len, d_model)  # absolute part
        # one separate embedding table for each period
        self.mods = nn.ModuleList([
            nn.Embedding(p, d_model) for p in periods
        ])
        self.periods = periods
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        pos = torch.arange(seq_len, device=x.device)  # [seq_len]
        # absolute PE
        pe = self.abs(pos)  # [seq_len, d_model]
        # add each periodic component
        for period, tbl in zip(self.periods, self.mods):
            pe = pe + tbl(pos % period)
        return x + pe.unsqueeze(0)


class TransformerClassifier(nn.Module):
    """
    Generic encoder-only transformer for sequence-level classification.
    """

    def __init__(
            self,
            max_len: int,  # The maximum possible NES sequence length
            embedding_dim: int,  # ESM embedding size
            positional_encoding: str,  # "sinusoidal" or "periodic_modulo"
            periods: tuple = None,  # Periods for the periodic encoding
            num_layers: int = 2,
            num_heads: int = 4,
            ff_dim: int = None,  # Dimension of the feed-forward at the end. If None → 4×embedding_dim
            dropout: float = 0.2,
            pooling: str = "cls",  # "cls" or "mean"
            add_cls_token: bool = True
    ):
        super().__init__()
        if positional_encoding not in ["sinusoidal", "periodic_modulo"]:
            raise ValueError("positional_encoding must be 'sinusoidal' or 'periodic_modulo'")
        if pooling not in ["cls", "mean"]:
            raise ValueError("pooling must be 'cls' or 'mean'")

        self.embedding_dim = embedding_dim
        self.pooling = pooling.lower()
        self.add_cls_token = add_cls_token

        if self.add_cls_token and self.pooling == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        if positional_encoding == "sinusoidal":
            if periods:
                raise ValueError("periods should not be specified for sinusoidal encodings!")
            self.pos_encoder = SinusoidalPositionalEncoding(
                d_model=embedding_dim, max_len=max_len + int(add_cls_token)
            )
        elif positional_encoding == "periodic_modulo":
            if periods is None:
                raise ValueError("periods must be specified for periodic modulo encodings!")
            self.pos_encoder = PeriodicModuloEncoding(
                d_model=embedding_dim, max_len=max_len + int(add_cls_token), periods=periods
            )

        ff_dim = ff_dim or embedding_dim * 4
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,  # (batch, seq, feature) in ≥1.13
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.norm = nn.LayerNorm(embedding_dim)
        self.classifier = nn.Linear(embedding_dim, 1)

        self.apply(_init_weights)

    def forward(
            self,
            x: torch.Tensor,
            key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x: [batch, seq_len, embedding_dim]  -- pre-computed embeddings
        key_padding_mask: [batch, seq_len] with True for PAD tokens
        """
        if self.add_cls_token and self.pooling == "cls":
            cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)  # prepend so that the input is [cls, embedding1, embedding2, ...]
            if key_padding_mask is not None:
                pad = torch.zeros(
                    (key_padding_mask.size(0), 1),
                    dtype=torch.bool,
                    device=key_padding_mask.device,
                )
                key_padding_mask = torch.cat([pad, key_padding_mask], dim=1)

        x = self.pos_encoder(x)  # Add positional encoding to input
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)

        if self.pooling == "cls":
            pooled = x[:, 0]  # [batch, dim]
        elif self.pooling == "mean":
            # ignore padding positions if mask supplied
            if key_padding_mask is not None:
                lens = (~key_padding_mask).sum(dim=1, keepdim=True)  # [batch,1]
                pooled = (x * (~key_padding_mask).unsqueeze(-1)).sum(dim=1) / lens
            else:
                pooled = x.mean(dim=1)
        else:
            raise ValueError("pooling must be 'cls' or 'mean'")
        # return (self.classifier(pooled)*0+0.5)+torch.tensor([0.,1.]).to("cuda")

        return self.classifier(pooled).squeeze(1)  # Tensor [batch_size, num_classes]


def get_transformer_classifier(max_seq_len: int, esm_embedding_dim: int,
                               num_layers: int = 2, num_heads: int = 4, dropout: float = 0.5,
                               positional_encoding: str = "sinusoidal", ) -> TransformerClassifier:
    """
    Creates a Transformer classifier for NES classification.
    :return: An instance of TransformerClassifier.
    """
    if positional_encoding == "sinusoidal":
        periods = None  # No periods needed for sinusoidal encoding
    else:
        periods = (2, 3, 4)

    return TransformerClassifier(
        max_len=max_seq_len,  # Maximum length of NES sequences
        embedding_dim=esm_embedding_dim,  # ESM-2 embedding size
        positional_encoding=positional_encoding,
        periods=periods,  # Known NES periods
        num_layers=num_layers,  # Number of transformer layers
        num_heads=num_heads,  # Number of attention heads
        ff_dim=None,  # Feed-forward dimension (default is 4×embedding_dim)
        dropout=dropout,  # Dropout rate
        pooling="cls",  # Use CLS token for pooling
        add_cls_token=True  # Add a CLS token to the input
    )

def main():
    """ Example usage """
    max_seq_len = 30  # Maximum length of NES sequences
    esm_embedding_dim = 320  # ESM-2 embedding size
    transformer_classifier = get_transformer_classifier(
        max_seq_len=max_seq_len,
        esm_embedding_dim=esm_embedding_dim,
    )
    return transformer_classifier
