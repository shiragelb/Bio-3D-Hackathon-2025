import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from Ex4_files.esm_embeddings import get_esm_model, get_esm_embeddings
from transformer_NES_classifier import TransformerClassifier  


def extract_nes_embeddings_from_csv(
    csv_path,
    name_col="name",
    sequence_col="full sequence",
    start_col="start#",
    nes_col="NES sequence",
    label_col="positive",
    embedding_size=320,
    embedding_layer=6,
    max_rows=10
):
    df = pd.read_csv(csv_path).head(max_rows)

    pep_tuples = list(zip(df[name_col], df[sequence_col]))

    esm_model, alphabet, batch_converter, device = get_esm_model(embedding_size=embedding_size)

    all_embeddings = get_esm_embeddings(
        pep_tuples,
        esm_model,
        alphabet,
        batch_converter,
        device,
        embedding_layer=embedding_layer,
        sequence_embedding=False
    )

    nes_embeddings = []
    for i, emb in enumerate(all_embeddings):
        start = df.iloc[i][start_col]
        nes_len = len(df.iloc[i][nes_col])
        nes_emb = emb[start:start + nes_len]
        nes_embeddings.append(torch.tensor(nes_emb, dtype=torch.float32))

    labels = torch.tensor(df[label_col].values, dtype=torch.long)
    return nes_embeddings, df, labels


def run(
    csv_path,
    name_col="name",
    sequence_col="full sequence",
    start_col="start#",
    nes_col="NES sequence",
    label_col="positive",
    id_col="uniprotID",
    embedding_size=320,
    embedding_layer=6,
    max_rows=10
):
    nes_embeddings, df, labels = extract_nes_embeddings_from_csv(
        csv_path,
        name_col=name_col,
        sequence_col=sequence_col,
        start_col=start_col,
        nes_col=nes_col,
        label_col=label_col,
        embedding_size=embedding_size,
        embedding_layer=embedding_layer,
        max_rows=max_rows
    )

    padded_embeddings = pad_sequence(nes_embeddings, batch_first=True)

    model = TransformerClassifier(
        max_len=padded_embeddings.size(1),
        embedding_dim=embedding_size,
        positional_encoding="periodic_modulo",
        periods=(2, 3, 4),
        num_classes=2,
        pooling="cls",
        add_cls_token=True
    )

    with torch.no_grad():
        logits = model(padded_embeddings)
        predictions = torch.argmax(logits, dim=1)

    return logits, predictions, df, labels, id_col


def save_predictions_to_csv(
    logits,
    predictions,
    df,
    id_col,
    label_col="positive",
    csv_output_path="model_output.csv"
):
    logits_str = [str(list(logit.detach().numpy())) for logit in logits]

    output_df = pd.DataFrame({
        id_col: df[id_col],
        "logits": logits_str,
        "predictions": predictions.tolist(),
        "labels": df[label_col].tolist()
    })

    output_df.to_csv(csv_output_path, index=False)


if __name__ == "__main__":
    logits, predictions, df, labels, id_col = run(
        csv_path="input_sequences/NESdb_NESpositive_sequences.csv",
        name_col="name",
        sequence_col="full sequence",
        start_col="start#",
        nes_col="NES sequence",
        label_col="positive",
        id_col="uniprotID",
        max_rows=10
    )

    save_predictions_to_csv(
        logits,
        predictions,
        df,
        id_col=id_col,
        label_col="positive",
        csv_output_path="model_output.csv"
    )
