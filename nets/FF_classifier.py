import torch
import torch.nn as nn


class SimpleDenseNet(nn.Module):
    """
    Very small fully-connected classifier for sequence ESM embeddings.

    Architecture
    ------------
    [emb_dim] -> Linear -> ReLU -> Dropout -> Linear -> ReLU -> Dropout -> Linear -> logit
    """

    def __init__(self, esm_emb_dim: int = 1280, hidden_dim: int = 128, dropout: float = 0.5):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(esm_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim // 2, 1)  # no sigmoid -> we use BCEWithLogitsLoss
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape [batch, emb_dim] containing float32 embeddings.

        Returns
        -------
        torch.Tensor
            Shape [batch] – raw logits (un-normalised scores).
        """
        return self.net(x.flatten(start_dim=1)).squeeze(1)

    def predict_on_test(self, x: torch.Tensor, W: int):
        """
        x: [batch, seq_len, embedding_dim]  – full-length test-set embeddings
        W: sliding-window length
        Returns  : list  (len == batch)
                   each entry is a 1-D tensor of window-level probabilities
                   shape [num_windows] where num_windows = max(seq_len-W+1, 0)
        """
        self.eval()  # inference mode
        device = next(self.parameters()).device  # model’s device
        x = x.to(device)
        all_logits, all_probs = [], []
        with torch.no_grad():
            idx = 0
            while idx + W <= x.size(1):
                logit = self(x[:, idx:idx + W, :])  # [B, 1]
                prob = torch.sigmoid(logit)  # [B, 1]
                all_logits.append(logit)
                all_probs.append(prob)
                idx += 1
        return all_logits, all_probs


def get_FF_classifier(seq_len: int, emb_dim: int):
    hidden_dim = 128
    dropout = 0.5
    return SimpleDenseNet(esm_emb_dim=emb_dim*seq_len, hidden_dim=hidden_dim, dropout=dropout)
