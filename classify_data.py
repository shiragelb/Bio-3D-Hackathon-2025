import os
import numpy as np
import pandas as pd, requests
import torch
from torch.nn.utils.rnn import pad_sequence

from Ex4_files.clustering import tsne_dim_reduction, kmeans_clustering
from Ex4_files.plot import plot_2dim_reduction
from pipeline import extract_embeddings, process_and_train
from train_model import data_to_loaders

rng = np.random.default_rng(42)

def predict_val_only(model, val_loader):
    device = next(model.parameters()).device
    model.eval()
    labels = []
    probabilities = []
    predictions = []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)  # [B, 2]
            prob = torch.softmax(logits, dim=1)  # [B, 2]
            preds = logits.argmax(dim=1)  # [B]
            # Storing predictions, probabilities, labels
            labels.extend(batch_y.cpu().numpy())  # [B]
            probabilities.extend(prob.cpu().numpy())  # [B, 2]
            predictions.extend(preds.cpu().numpy())  # [B]
    return np.array(predictions), np.array(probabilities), np.array(labels)


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


def sample_20mer_and_start(seq: str, win: int = 20, rng=rng):
    """
    Return (subseq, start_idx).  If the sequence is shorter than `win`,
    keep it unchanged and report start_idx = 0.
    """
    L = len(seq)
    if L <= win:
        return seq, 0
    start = int(rng.integers(0, L - win + 1))   # 0 … L-win (inclusive)
    return seq[start : start + win], start

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
    uids = df["uniprotID"]
    # Fetch sequences for each UniProt ID, and add to a new column "sequence":
    df["full sequence"] = uids.apply(fetch_uniprot_fasta)
    df[["NOT NES", "start#"]] = (
        df["full sequence"]
        .apply(sample_20mer_and_start)  # -> Series of tuples
        .apply(pd.Series)  # -> two separate columns
    )
    df = df.dropna(subset=["full sequence"])  # Drop rows with no sequence
    # draw a random subsequence of len 20 that still fits inside the sequence.
    df.to_csv(neg_csv_path_new, index=False)
    return neg_csv_path_new


def plot_embeds_in_2d(embeds, labels, id=""):
    train_np = np.array([emb.mean(0) for emb in embeds])
    # labels are currently in [0/1, 0/1] format, convert to 0/1:
    labels = [1 if label[1] == 1 else 0 for label in labels]
    k_means_labels = kmeans_clustering(train_np, k=2)
    coords_2d = tsne_dim_reduction(train_np, dim=2)

    id = "_" + id if id else ""
    print("Plotting 2D dimensionality reduction by true labels and by K-means clustering")
    plot_2dim_reduction(coords_2d, [["N", "P"][i] for i in labels], out_file_path=f"2d_true_labels{id}.png")
    plot_2dim_reduction(coords_2d, k_means_labels, out_file_path=f"2d_k_means_{id}.png")


def classify_difficulty():
    pos_csv_path = "input_sequences/NESdb_NESpositive_sequences.csv"
    neg_csv_path = create_neg_csv()  # Create a negative CSV file with random subsequences
    os.makedirs("embeddings", exist_ok=True)
    pos_embeds, pos_labels, _, _ = extract_embeddings(pos_csv_path, embedding_path="train_embeddings.pt")
    neg_embeds, labels, _, device = extract_embeddings(neg_csv_path,
                                                       embedding_path="embeddings/neg_for_classification.pt")
    plot_embeds_in_2d(neg_embeds, labels, id="negative_embeddings")

    # First, train a model on the embeddings:
    neg_embeds = pad_sequence(neg_embeds, batch_first=True)
    max_seq_len = neg_embeds.size(1)
    emb_dim = neg_embeds.size(2)

    cutoff = int(0.5 * len(neg_embeds))  # Split data at 50%
    neg_embeds_1 = neg_embeds[:cutoff]
    labels_1 = labels[:cutoff]
    neg_embeds_2 = neg_embeds[cutoff:]
    labels_2 = labels[cutoff:]

    train_loader1 = data_to_loaders(neg_embeds_1 + pos_embeds, labels_1 + pos_labels, train_fraction=1)
    classify_loader1 = data_to_loaders(neg_embeds_2, labels_2, train_fraction=1)
    hard_indices_2 = _classify_difficulty(classify_loader1, train_loader1, max_seq_len, emb_dim)

    train_loader2 = data_to_loaders(neg_embeds_2 + pos_embeds, labels_2 + pos_labels, train_fraction=1)
    classify_loader2 = data_to_loaders(neg_embeds_1, labels_1, train_fraction=1)
    hard_indices_1 = _classify_difficulty(train_loader2, classify_loader2, max_seq_len, emb_dim)

    hard_neg_embeds = np.concatenate(
        [neg_embeds_1[i] for i in hard_indices_1] + [neg_embeds_2[i] for i in hard_indices_2])
    hard_labels = np.concatenate([labels_1[i] for i in hard_indices_1] + [labels_2[i] for i in hard_indices_2])

    assert len(hard_labels) == len(hard_neg_embeds)
    print(f"Found {len(hard_neg_embeds)} hard negative examples.")

    os.makedirs("embeddings", exist_ok=True)
    torch.save({
        'nes_embeddings': hard_neg_embeds,
        'labels': hard_labels,
        'device': device
    }, "embeddings/hard_negative_embeddings.pt")


if __name__ == '__main__':
    classify_difficulty()
