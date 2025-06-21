import numpy as np
import pandas as pd
import requests
import os

import torch
from torch.nn.utils.rnn import pad_sequence

from Ex4_files.esm_embeddings import get_esm_model, get_esm_embeddings

rng_ = np.random.default_rng(42)

EMBED_SIZE = 2560
EMBED_LAYER = 9


def extract_embeddings(csv_path, embedding_path: str, embedding_size=320, embedding_layer=6):
    if os.path.exists(embedding_path):
        print("Loading embeddings from", embedding_path)
        embeddings_data = torch.load(embedding_path, weights_only=False)
        return embeddings_data['embeddings'], embeddings_data['labels'], embeddings_data['df'], embeddings_data[
            'device']

    print("Extracting embeddings from", csv_path)
    df = pd.read_csv(csv_path)
    pep_tuples = list(zip(df["uniprotID"], df["full sequence"]))

    esm_model, alphabet, batch_converter, device = get_esm_model(embedding_size=embedding_size)

    embed_batch_size = 1  # Adjust batch size, according to your GPU memory. 6GB VRam appears to handle 8.
    nes_embeddings = []
    total = len(pep_tuples)
    current = 0
    bad_labels = []
    while current < total:
        if current + embed_batch_size > total:  # Last batch may be smaller
            embed_batch_size = total - current
        print(f"Processing embedding {current + embed_batch_size}/{total}...")
        batch = pep_tuples[current:current + embed_batch_size]

        # Get embeddings for the current batch
        batch_embeddings = get_esm_embeddings(
            batch,
            esm_model,
            alphabet,
            batch_converter,
            device,
            embedding_layer=embedding_layer,
            sequence_embedding=False
        )

        for i, emb in enumerate(batch_embeddings):
            i = i + current
            start = df.iloc[i]["start#"]
            try:
                if df.iloc[i]["label"] == 1:
                    length = len(df.iloc[i]["NES sequence"])
                else:
                    length = len(df.iloc[i]["NOT NES"])
                nes_emb = emb[start:start + length]
                nes_embeddings.append(torch.tensor(nes_emb, dtype=torch.float32))
            except TypeError:
                print("error with sequence at index", i)
                bad_labels.append(i)

        current += embed_batch_size

    df.drop(index=bad_labels, inplace=True)  # drop rows with bad labels
    # convert labels to tensor of shape (N, 2):
    labels = torch.tensor(df["label"].values, dtype=torch.float32)
    torch.save({
        'embeddings': nes_embeddings,
        'labels': labels,
        'df': df,
        'device': device
    }, embedding_path)
    return nes_embeddings, labels, df, device


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


def create_neg_proteom_csv() -> tuple[str, str]:
    print("Creating negative proteomics CSV files...")
    neg_csv_path = "DB_Tanya/deep_proteomics_data_3.csv"
    neg_train_csv = "input_sequences/negative_nes_proteomics3_train.csv"
    neg_test_csv = "input_sequences/negative_nes_proteomics3_test.csv"

    if os.path.exists(neg_train_csv) and os.path.exists(neg_test_csv):
        return neg_train_csv, neg_test_csv

    df = pd.read_csv(neg_csv_path)
    # Filter to rows when col "Category of binding" is "Non-binder"
    df = df[df["Category of binding"] == "Non-binder"]
    # rename the column UNIPORT to uniprotID:
    df = df.rename(columns={"UNIPROT": "uniprotID"})
    # Check for duplicates in "uniprotID" column:
    if df["uniprotID"].duplicated().any():
        print("Warning: There are duplicate UniProt IDs in the dataset. ")
        raise ValueError("Duplicate UniProt IDs found in the dataset.")
    uids = df["uniprotID"]
    # Fetch sequences for each UniProt ID, and add to a new column "sequence":
    df["full sequence"] = uids.apply(fetch_uniprot_fasta)
    df = df.drop_duplicates(subset=["full sequence"])  # Delete rows with duplicate sequences:
    if df["full sequence"].duplicated().any():
        print("Warning: There are sequences in the dataset. ")
        raise ValueError("Duplicate sequences found in the dataset.")
    # Drop empty sequences:
    df = df.dropna(subset=["full sequence"])
    # Fetch random sub-sequence of length 20 and its start index:
    df[["NOT NES", "start#"]] = (
        df["full sequence"]
        .apply(sample_20mer_and_start)  # -> Series of tuples
        .apply(pd.Series)  # -> two separate columns
    )
    df["label"] = 0

    # Split the dataset into train and test sets:
    train_size = int(0.6 * len(df))
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]

    train_df.to_csv(neg_train_csv, index=False)
    test_df.to_csv(neg_test_csv, index=False)
    print(f"Negative proteomics CSV files created: {neg_train_csv}, {neg_test_csv}")
    return neg_train_csv, neg_test_csv


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


def create_pos_proteom_csv():
    """
    Create a CSV file with positive proteomics data.
    Note that since we don't have the specific NES sequence, this data is only viable for testing.
    """
    print("Creating positive proteomics CSV files...")
    csv_path = "DB_Tanya/deep_proteomics_data_3.csv"
    pos_test_csv = "input_sequences/pos_test_proteomics3.csv"
    if os.path.exists(pos_test_csv):
        return pos_test_csv

    positives = ["Cargo A", "Cargo B", "Low Abundant Cargo"]
    df = pd.read_csv(csv_path)
    # rename the column UNIPORT to uniprotID:
    df = df.rename(columns={"UNIPROT": "uniprotID"})
    # Filter to rows when col "Category of binding" is in positives:
    df = df[df["Category of binding"].isin(positives)]
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
    df["label"] = 1

    # Split the dataset into train and test sets:
    df.to_csv(pos_test_csv, index=False)
    print(f"Positive proteomics CSV file created: {pos_test_csv}")
    return pos_test_csv


def combine_csvs(csvs_list: list[str], out_path: str):
    """
    Create a combined CSV file from a list of CSV files.
    """
    combined_df = pd.concat([pd.read_csv(csv) for csv in csvs_list], ignore_index=True)
    combined_df.to_csv(out_path, index=False)
    return out_path


def extract_train_embeddings(train_csv_path, embedding_path):
    embeds, labels, _, _ = extract_embeddings(train_csv_path, embedding_path=embedding_path,
                                              embedding_size=EMBED_SIZE, embedding_layer=EMBED_LAYER)
    # Look for the maximum length of the sequences in the dataset, across all embeddings:
    embeds = pad_sequence(embeds, batch_first=True)
    return embeds, labels


def extract_test_embeddings(test_csv_path: str, embedding_path: str):
    if os.path.exists(embedding_path):
        print(f"Test embeddings already exist at {embedding_path}.")
        data = torch.load(embedding_path, weights_only=False)
        embeddings = data['embeddings']
        labels = data['labels']
        df = data['df']
        device = data['device']
        return embeddings, labels
    print(f"Extracting test embeddings from {test_csv_path} to {embedding_path}...")
    df = pd.read_csv(test_csv_path)
    pep_tuples = list(zip(df["uniprotID"], df["full sequence"]))
    esm_model, alphabet, batch_converter, device = get_esm_model(embedding_size=EMBED_SIZE)

    embed_batch_size = 1  # Adjust batch size, according to your GPU memory. 40GB VRam appears to handle 4.
    embeddings = []
    total = len(pep_tuples)
    current = 0
    while current < total:
        if current + embed_batch_size > total:  # Last batch may be smaller
            embed_batch_size = total - current
        print(f"Processing embedding {current + embed_batch_size}/{total}...")
        batch = pep_tuples[current:current + embed_batch_size]

        # Get embeddings for the current batch
        batch_embeddings = get_esm_embeddings(
            batch,
            esm_model,
            alphabet,
            batch_converter,
            device,
            embedding_layer=EMBED_LAYER,
            sequence_embedding=False
        )
        for emb in batch_embeddings:
            embeddings.append(torch.tensor(emb, dtype=torch.float32))

        current += embed_batch_size

    labels = torch.tensor(df["label"].values, dtype=torch.float32)
    ids = df["uniprotID"].values
    torch.save({
        'embeddings': embeddings,
        'labels': labels,
        'uniprotIDs': ids,
        'df': df,
        'device': device
    }, embedding_path)
    print("Test embeddings created: ", embedding_path)
    return embeddings, labels


def create_NESdb_csv(pos_train__path: str, out_path: str):
    df = pd.read_csv(pos_train__path)
    df.drop(columns=["species", "name"], inplace=True)
    df.to_csv(out_path, index=False)
    return out_path


def create_data():
    os.makedirs("input_sequences_processed", exist_ok=True)
    pos_train_path = "input_sequences/NESdb_NESpositive_sequences.csv"
    pos_csv_train_path = create_NESdb_csv(pos_train_path, out_path="input_sequences_processed/NESdb_pos_train_set.csv")
    neg_csv_train_path, neg_csv_test_path = create_neg_proteom_csv()
    pos_csv_test_path = create_pos_proteom_csv()

    train_set = [neg_csv_train_path, pos_csv_train_path]
    test_set = [neg_csv_test_path, pos_csv_test_path]

    full_train_csv = combine_csvs(train_set, out_path="input_sequences_processed/full_train_set.csv")
    full_test_csv = combine_csvs(test_set, out_path="input_sequences_processed/full_test_set.csv")

    os.makedirs("embeddings", exist_ok=True)
    train_path = F"embeddings/full_train_embeddings_{EMBED_SIZE}_{EMBED_LAYER}.pt"
    train_embeds, train_labels = extract_train_embeddings(full_train_csv,
                                                          embedding_path=train_path)
    test_path = f"embeddings/full_test_embeddings_{EMBED_SIZE}_{EMBED_LAYER}.pt"
    test_embeds, test_labels = extract_test_embeddings(full_test_csv,
                                                       embedding_path=test_path)

    return train_embeds, train_labels, test_embeds, test_labels


if __name__ == '__main__':
    create_data()
    print("Data creation completed.")
