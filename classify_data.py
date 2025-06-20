import os
import numpy as np
import pandas as pd, requests
import torch
from torch.nn.utils.rnn import pad_sequence
from Ex4_files.clustering import tsne_dim_reduction, kmeans_clustering
from Ex4_files.plot import plot_2dim_reduction
from pipeline import extract_embeddings, process_and_train
from train_model import data_to_loaders

rng_ = np.random.default_rng(42)

def predict_val_only(model, val_loader, threshold: float = 0.5):
    """
    Run the model on `val_loader` and return:
        preds  – ndarray shape [N] of 0/1 class predictions
        probs  – ndarray shape [N, 2] with [p_neg, p_pos] per sample
        labels – ndarray shape [N] of ground-truth class indices
    """
    device = next(model.parameters()).device
    model.eval()

    preds, probs_all, labels = [], [], []

    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)              # yb: 0/1 or one-hot

            logits = model(xb)                                 # [B] or [B,1]
            p_pos  = torch.sigmoid(logits).squeeze(dim=-1)     # [B]
            p_neg  = 1.0 - p_pos
            preds_batch = (p_pos >= threshold).long()          # [B] – 0 or 1

            # ── book-keeping ───────────────────────────────────────────────
            labels.extend( (yb.argmax(dim=1) if yb.dim() == 2 else yb)
                           .cpu().numpy() )
            preds.extend(preds_batch.cpu().numpy())
            probs_all.extend( torch.stack([p_neg, p_pos], dim=1).cpu().numpy() )

    return (np.asarray(preds,      dtype=np.int64),
            np.asarray(probs_all,  dtype=np.float32),
            np.asarray(labels,     dtype=np.int64))


def _classify_difficulty(train_loader, classify_loader, max_seq_len, emb_dim):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = process_and_train(train_loader=train_loader, val_loader=classify_loader, device=device,
                              max_seq_len=max_seq_len, emb_dim=emb_dim)

    # Now, classify the difficulty of each sequence:
    predictions, probs, true_labels = predict_val_only(model, classify_loader)
    # Find hard examples:
    prob_positive = probs[:, 1]  # P(class==1) for each sample

    thr_low = 0.25  # Threshold for hard positives
    thr_high = 0.75  # Threshold for hard negatives

    pos_hard = np.where((true_labels == 1) & (prob_positive < thr_low))[0]
    neg_hard = np.where((true_labels == 0) & (prob_positive > thr_high))[0]

    hard_indices = np.concatenate([pos_hard, neg_hard])
    return hard_indices


def fetch_uniprot_fasta(uid):
    """Return the primary sequence for a single UniProt accession or None if unavailable."""
    if pd.isna(uid):
        return None
    url = 'https://rest.uniprot.org/uniprotkb/' + uid + '.fasta'
    try:
        r = requests.get(url, headers={'Accept': 'text/plain'}, timeout=10)
        if r.status_code != 200 or not r.text.startswith('>'):
            return None  # no FASTA returned
        # strip header line and join the remaining lines
        return ''.join(r.text.split('\n')[1:]).strip()
    except Exception:
        return None


def sample_20mer_and_start(seq: str, win: int = 20, rng=rng_):
    """
    Return (subseq, start_idx).  If the sequence is shorter than `win`,
    keep it unchanged and report start_idx = 0.
    """
    L = len(seq)
    if L <= win:
        return seq, 0
    start = int(rng.integers(0, L - win + 1))  # 0 … L-win (inclusive)
    return seq[start: start + win], start


def create_neg_csv() -> str:
    neg_csv_path = "DB_Tanya/deep_proteomics_data_3.csv"
    neg_csv_path_new = "input_sequences/negative_nes_proteomics3.csv"

    if os.path.exists(neg_csv_path_new):
        return neg_csv_path_new

    df = pd.read_csv(neg_csv_path)
    # Filter to rows when col "Category of binding" is "Non-binder"
    df = df[df["Category of binding"] == "Non-binder"]
    # rename the column UNIPORT to uniprotID:
    df = df.rename(columns={"UNIPROT": "uniprotID"})
    # Check for duplicates in "uniprotID" column:
    if df["uniprotID"].duplicated().any():
        print("Warning: There are duplicate UniProt IDs in the dataset. "
              "This may lead to unexpected results when fetching sequences.")
        raise ValueError("Duplicate UniProt IDs found in the dataset.")

    uids = df["uniprotID"]
    # Fetch sequences for each UniProt ID, and add to a new column "sequence":
    df["full sequence"] = uids.apply(fetch_uniprot_fasta)
    # Drop empty sequences:
    df = df.dropna(subset=["full sequence"])
    # Fetch random sub-sequence of length 20 and its start index:
    df[["NOT NES", "start#"]] = (
        df["full sequence"]
        .apply(sample_20mer_and_start)  # -> Series of tuples
        .apply(pd.Series)  # -> two separate columns
    )
    df["label"] = 0

    # draw a random subsequence of len 20 that still fits inside the sequence.
    df.to_csv(neg_csv_path_new, index=False)
    return neg_csv_path_new


def plot_embeds_in_2d(embeds, labels, id=""):
    train_np = np.array([emb.mean(0) for emb in embeds])
    k_means_labels = kmeans_clustering(train_np, k=2)
    coords_2d = tsne_dim_reduction(train_np, dim=2)

    id = "_" + id if id else ""
    print("Plotting 2D dimensionality reduction by true labels and by K-means clustering")
    print(labels.shape)
    plot_2dim_reduction(coords_2d, [["N", "P"][int(label.item())] for label in labels], out_file_path=f"2d_true_labels{id}.png")
    plot_2dim_reduction(coords_2d, k_means_labels, out_file_path=f"2d_k_means_{id}.png")


def pad_first(seqs, target_len: int = 20, pad_value: int | float = 0):
    """
    Pad the *first* tensor in a list/tuple to `target_len` tokens
    (along dim-0) if it is shorter.  Works for 1-D or 2-D tensors.
    Returns the same container so your later code stays the same.
    """
    first = seqs[0]

    if first.size(0) > target_len:
        raise ValueError(
            f"First sequence length ({first.size(0)}) exceeds target_len ({target_len})."
        )

    pad_len = target_len - first.size(0)
    if pad_len:
        # Preserve any extra feature dims, e.g. [L, d]
        pad_shape = (pad_len, *first.shape[1:])
        pad_tensor = first.new_full(pad_shape, pad_value)
        seqs[0] = torch.cat((first, pad_tensor), dim=0)

    return seqs  # optional, but convenient


def get_embeds(pos_csv_path, neg_csv_path):
    pos_embeds, pos_labels, _, _ = extract_embeddings(pos_csv_path, embedding_path="train_embeddings.pt")
    neg_embeds, neg_labels, _, _ = extract_embeddings(neg_csv_path,
                                                           embedding_path="embeddings/neg_for_classification.pt")
    pad_first(pos_embeds, target_len=20)
    pos_embeds = pad_sequence(pos_embeds, batch_first=True)
    neg_embeds = pad_sequence(neg_embeds, batch_first=True)

    return pos_embeds, pos_labels, neg_embeds, neg_labels


def classify_difficulty():
    pos_csv_path = "input_sequences/NESdb_NESpositive_sequences.csv"
    neg_csv_path = create_neg_csv()  # Create a negative CSV file with random subsequences
    os.makedirs("embeddings", exist_ok=True)
    pos_embeds, pos_labels, neg_embeds, neg_labels = get_embeds(pos_csv_path, neg_csv_path)
    # Plotting embeddings after reduction to 2D:
    plot_embeds_in_2d(neg_embeds, neg_labels, id="negative_embeddings")
    # Prepare embeddings:
    max_seq_len = 20
    # print("neg_embeds shape ", neg_embeds[0].shape)
    # print("pos_embeds shape ", pos_embeds[0].shape)


    # print("pos_embeds shape ", pos_embeds[0].shape)
    print(f"There are {len(pos_embeds)} positive embeddings.")
    print(f"There are {len(neg_embeds)} negative embeddings.")

    emb_dim = neg_embeds.size(2)

    cutoff = int(0.5 * len(neg_embeds))  # Split data at 50%
    neg_embeds_1 = neg_embeds[:cutoff]
    labels_1 = neg_labels[:cutoff]
    neg_embeds_2 = neg_embeds[cutoff:]
    labels_2 = neg_labels[cutoff:]
    assert len(neg_embeds_1) == len(labels_1)
    assert len(neg_embeds_2) == len(labels_2)

    # Train models:
    embeds1 = torch.cat((neg_embeds_1, pos_embeds), dim=0)
    labels1 = torch.cat((labels_1, pos_labels), dim=0)

    embeds2 = torch.cat((neg_embeds_2, pos_embeds), dim=0)
    labels2 = torch.cat((labels_2, pos_labels), dim=0).to(dtype=torch.int)

    train_loader1 = data_to_loaders(embeds1, labels1, train_fraction=1)
    classify_loader1 = data_to_loaders(neg_embeds_2, labels_2, train_fraction=1)
    hard_indices_2 = _classify_difficulty(train_loader1, classify_loader1, max_seq_len, emb_dim)

    train_loader2 = data_to_loaders(embeds2, labels2, train_fraction=1)
    classify_loader2 = data_to_loaders(neg_embeds_1, labels_1, train_fraction=1)
    hard_indices_1 = _classify_difficulty(train_loader2, classify_loader2, max_seq_len, emb_dim)

    # print(len(hard_indices_1))
    # print(len(neg_embeds_1))
    # print(len(hard_indices_2))
    # print(len(neg_embeds_2))
    #
    # hard_neg_embeds = np.concatenate(
    #     [neg_embeds_1[i] for i in hard_indices_1] + [neg_embeds_2[i] for i in hard_indices_2])
    # hard_labels = np.concatenate([labels_1[i] for i in hard_indices_1] + [labels_2[i] for i in hard_indices_2])

    # assert len(hard_labels) == len(hard_neg_embeds)
    # print(f"Found {len(hard_neg_embeds)} hard negative examples.")
    #
    # os.makedirs("embeddings", exist_ok=True)
    # save_path = "embeddings/hard_negative_embeddings.pt"
    # torch.save({
    #     'nes_embeddings': hard_neg_embeds,
    #     'labels': hard_labels,
    #     'device': device
    # }, save_path)
    # print("Saved to: ", save_path)


if __name__ == '__main__':
    classify_difficulty()
