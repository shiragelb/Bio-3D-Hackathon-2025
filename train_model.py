import numpy as np
import torch.nn
from torch import optim, nn
from torch.utils.data import DataLoader, random_split, TensorDataset
from tqdm.auto import tqdm
from pathlib import Path


def accuracy(logits:torch.Tensor, y):
    """Compute categorical accuracy."""
    # print("logits: ", logits[:8])
    preds = logits.argmax(dim=1)
    gt = y.argmax(dim=1)
    return (preds == gt).float().mean().item()


def data_to_loaders(x, y):
    dataset = TensorDataset(x, y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # <- reproducible split
    )

    # Step 4:
    BATCH_SIZE = 16
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,  # shuffle only the training set
        drop_last=False  # keep last partial batch
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    return train_loader, val_loader


def train(model, train_loader, val_loader, device):
    # ───────────────────────────── config ────────────────────────────────
    N_EPOCHS = 20
    LR = 1e-6
    WEIGHT_DECAY = 1e-4
    PATIENCE = 99  # early-stopping patience (epochs)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    # ─────────────────────────── training loop ───────────────────────────
    best_acc = 0.0
    epochs_no_improve = 0
    save_path = Path("best_model.pt")

    for epoch in range(1, N_EPOCHS + 1):
        # ----- train phase -----
        model.train()
        train_loss, train_acc = 0.0, 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{N_EPOCHS} [train]", leave=False)
        for batch_x, batch_y in pbar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)  # forward pass
            loss = criterion(logits, batch_y)  # CE loss
            loss.backward()  # backward pass (FP32 grads)
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)
            with torch.no_grad():
                # compute accuracy
                train_acc += accuracy(logits, batch_y) * batch_x.size(0)

        train_loss /= len(train_loader.dataset)
        train_acc /= len(train_loader.dataset)

        # ----- validation phase -----
        model.eval()
        val_loss, val_acc = 0.0, 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)

                val_loss += loss.item() * batch_x.size(0)
                val_acc += accuracy(logits, batch_y) * batch_x.size(0)

        val_loss /= len(val_loader.dataset)
        val_acc /= len(val_loader.dataset)

        # scheduler.step(val_acc)  # adjust LR if plateau

        tqdm.write(
            f"Epoch {epoch:02d} | "
            f"train loss/acc: {train_loss:.4f}/{train_acc:.3f} | "
            f"val loss/acc: {val_loss:.4f}/{val_acc:.3f}"
        )

        # ----- early stopping & checkpoint -----
        if val_acc > best_acc:
            best_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            tqdm.write(f"  ↳ New best val-acc {best_acc:.3f} → saved to {save_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                tqdm.write("Early stopping triggered.")
                break

    print(f"Training complete. Best validation accuracy: {best_acc:.3f}")


def evaluate(model, val_loader, device):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    val_loss, val_acc = 0.0, 0.0

    labels = []
    probabilities = []
    predictions = []

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)


            logits = model(batch_x)                    # [B, 2]
            prob   = torch.softmax(logits, dim=1)      # [B, 2]
            preds  = logits.argmax(dim=1)              # [B]

            # Storing predictions, probabilities, labels
            labels.extend(batch_y.cpu().numpy())       # [B]
            probabilities.extend(prob.cpu().numpy())   # [B, 2]
            predictions.extend(preds.cpu().numpy())    # [B]

            loss = criterion(logits, batch_y)

            val_loss += loss.item() * batch_x.size(0)
            val_acc += accuracy(logits, batch_y) * batch_x.size(0)

    val_loss /= len(val_loader.dataset)
    val_acc /= len(val_loader.dataset)
    return val_loss, val_acc, np.array(predictions), np.array(probabilities), np.array(labels)
