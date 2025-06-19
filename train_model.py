import numpy as np
import torch.nn
from torch import optim, nn
from torch.utils.data import DataLoader, random_split, TensorDataset
from tqdm.auto import tqdm
from pathlib import Path
import torch


def accuracy(logits: torch.Tensor, y):
    """Compute categorical accuracy."""
    # print("logits: ", logits[:8])
    preds = logits.argmax(dim=1)
    gt = y.argmax(dim=1)
    return (preds == gt).float().mean().item()


def data_to_loaders(x, y, train_fraction=0.8, sampler=None):
    dataset = TensorDataset(x, y)
    train_size = int(train_fraction * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # <- reproducible split
    )

    BATCH_SIZE = 16
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True if not sampler else False,  # shuffle only the training set
        drop_last=False,  # keep last partial batch,
        sampler=sampler  # use sampler if provided
    )
    if train_fraction < 1:
        val_loader = DataLoader(
            val_ds,
            batch_size=BATCH_SIZE,
            shuffle=False
        )
        return train_loader, val_loader

    return train_loader


def train(model, train_loader, val_loader, device):
    # ───────────────────────────── config ────────────────────────────────
    N_EPOCHS = 30
    LR = 1e-5
    WEIGHT_DECAY = 1e-4
    PATIENCE = 99  # early-stopping patience (epochs)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    # ─────────────────────────── training loop ───────────────────────────
    best_acc = 0.0
    epochs_no_improve = 0
    save_path = Path("best_model.pt")
    all_train_preds = []
    for epoch in range(1, N_EPOCHS + 1):
        epoch_probs = []
        positive_labels = 0
        negative_labels = 0
        # ----- train phase -----
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{N_EPOCHS} [train]", leave=False)
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)

            positive_labels += (yb == 1).sum().item()  # count positive labels
            negative_labels += (yb == 0).sum().item()  # count negative labels

            optimizer.zero_grad()

            logits = model(xb)  # [B, ]
            loss = criterion(logits, yb.float())  # BCEWithLogits -> float targets

            loss.backward()
            optimizer.step()

            # --- bookkeeping ----------------------------------------------------
            batch_size = xb.size(0)
            train_samples += batch_size
            train_loss += loss.item() * batch_size

            with torch.no_grad():
                p_pos = torch.sigmoid(logits).squeeze(dim=-1)  # [B]
                p_neg = 1.0 - p_pos
                preds_batch = (p_pos >= 0.5).long()  # [B] – 0 or 1
                epoch_probs.extend(p_pos.cpu().numpy())
                all_train_preds.extend(preds_batch.cpu().numpy())

                # yb may be one-hot; turn into class indices if so
                y_idx = yb.argmax(dim=1) if yb.dim() == 2 else yb
                train_correct += (preds_batch == y_idx).sum().item()
            # --------------------------------------------------------------------

        avg_train_loss = train_loss / train_samples
        avg_train_acc = train_correct / train_samples  # 0–1 range
        # print(f"Positive labels: {positive_labels}, Negative labels: {negative_labels}")

        # ---------- validation phase ----------
        model.eval()
        val_loss     = 0.0
        val_correct  = 0
        val_samples  = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)

                logits = model(xb)
                loss   = criterion(logits, yb.float())
                bs            = xb.size(0)
                val_samples  += bs
                val_loss     += loss.item() * bs
                p_pos = torch.sigmoid(logits).squeeze(dim=-1)  # [B]
                preds_batch = (p_pos >= 0.5).long()  # [B] – 0 or 1
                y_idx         = yb.argmax(dim=1) if yb.dim() == 2 else yb
                val_correct  += (preds_batch == y_idx).sum().item()


        avg_val_loss = val_loss / val_samples
        avg_val_acc  = val_correct / val_samples

        tqdm.write(
            f"Epoch {epoch:02d} | "
            f"train loss/acc: {avg_train_loss:.4f}/{avg_train_acc:.3f} | "
            f"val loss/acc: {avg_val_loss:.4f}/{avg_val_acc:.3f} | "
            f"Avg probability during training epoch: {np.mean(epoch_probs):.3f} | "
            f"Variance of probabilities during training epoch: {np.var(epoch_probs):.3f}"
        )

        # ----- early stopping & checkpoint -----
        if avg_val_acc > best_acc:
            best_acc = avg_val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            # tqdm.write(f"  ↳ New best val-acc {best_acc:.3f} → saved to {save_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                tqdm.write("Early stopping triggered.")
                break

    print(f"Training complete. Best validation accuracy: {best_acc:.3f}")



def evaluate(model, val_loader, device):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    positive_labels = 0
    negative_labels = 0

    total_loss      = 0.0
    total_correct   = 0
    total_samples   = 0

    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)         # y: [B, 2] one-hot
            positive_labels += (y == 1).sum().item()  # count positive labels
            negative_labels += (y == 0).sum().item()  # count negative labels

            logits = model(x)                                 # [B] or [B,1]
            p_pos  = torch.sigmoid(logits).squeeze(dim=-1)     # [B]
            p_neg  = 1.0 - p_pos
            preds_batch = (p_pos >= 0.5).long()          # [B] – 0 or 1
            loss = criterion(logits, y.float())
            batch_size        = x.size(0)
            total_samples    += batch_size
            total_loss       += loss.item() * batch_size
            total_correct    += (preds_batch == y).sum().item()

            # store for later analysis
            all_preds.extend(preds_batch.cpu().numpy())
            all_probs.extend(p_pos.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / total_samples
    avg_acc  = total_correct / total_samples          # 0 – 1 range

    # print(f"Positive labels: {positive_labels}, Negative labels: {negative_labels}")
    return (avg_loss,
            avg_acc,
            np.array(all_preds),
            np.array(all_probs),
            np.array(all_labels))
