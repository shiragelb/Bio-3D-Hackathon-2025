import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from tqdm import tqdm
import random


def fetch_uniprot_fasta(uid):
    """Fetch full protein sequence from UniProt."""
    if pd.isna(uid):
        return None
    url = f'https://rest.uniprot.org/uniprotkb/{uid}.fasta'
    try:
        r = requests.get(url, headers={'Accept': 'text/plain'}, timeout=10)
        if r.status_code != 200 or not r.text.startswith('>'):
            return None
        return ''.join(r.text.split('\n')[1:]).strip()
    except Exception:
        return None


def fetch_nes_table_to_csv(url, output_csv_path="nes_output.csv"):
    response = requests.get(url)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table")
    headers = [th.text.strip() for th in table.find_all("th")]

    rows_data = []

    for row in table.find_all("tr")[1:]:  # skip header row
        cols = row.find_all("td")
        if not cols:
            continue

        sequence_td = cols[5]  # the NES sequence column

        clean_sequence = ""
        mutations = ""
        red_indices = []

        index = 0
        in_sequence = True

        for content in sequence_td.contents:
            if isinstance(content, str):
                text = content.strip()
                if text and any(c.isalpha() for c in text):
                    clean_sequence += text
                    index += len(text)
                elif text and not in_sequence:
                    mutations += text
            elif content.name == "font" and content.get("color") == "red":
                letter = content.get_text()
                clean_sequence += letter
                red_indices.append(index)
                index += 1
            elif content.name == "br":
                in_sequence = False
            elif not in_sequence and content.name is None:
                mutations += content.strip()

        row_values = []
        for i, col in enumerate(cols):
            if i == 5:
                row_values.append(clean_sequence)
            else:
                row_values.append(col.get_text(strip=True))

        row_values.append(len(clean_sequence))  # length
        row_values.append(mutations)           # mutations
        row_values.append(red_indices)         # places
        rows_data.append(row_values)

    headers[5] = "NES sequence"
    headers += ["length", "mutations", "places"]

    df = pd.DataFrame(rows_data, columns=headers)

    # Fetch full UniProt sequences
    print("Fetching UniProt full sequences …")
    unique_ids = df["uniprotID"].dropna().unique()
    seq_lookup = {}

    for uid in tqdm(unique_ids):
        seq_lookup[uid] = fetch_uniprot_fasta(uid)
        time.sleep(0.3)

    df["full sequence"] = df["uniprotID"].map(seq_lookup)

    # Add start# column
    df = add_start_column(df)

    df.to_csv(output_csv_path, index=False)
    print(f"[✓] Saved to {output_csv_path}")


def add_start_column(df):
    """
    Adds a 'start#' column to the dataframe by locating the NES sequence within the full sequence.
    """
    start_positions = []

    for _, row in df.iterrows():
        nes = row.get("NES sequence")
        full = row.get("full sequence")

        if pd.isna(nes) or pd.isna(full):
            start_positions.append(None)
            continue

        start_index = full.find(nes)
        start_positions.append(start_index if start_index != -1 else None)

    df["start#"] = start_positions
    return df


def generate_negative_csv_from_nes(nes_csv_path, output_csv_path=None, seed=42):
    """
    Generates a negative dataset by mutating 2 positions in the NES sequence
    based on the 'places' list and saves it to a new CSV.
    """
    random.seed(seed)
    df = pd.read_csv(nes_csv_path)

    negative_rows = []

    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")  # standard amino acids

    for _, row in df.iterrows():
        sequence = row["NES sequence"]
        places = eval(row["places"]) if isinstance(row["places"], str) else row["places"]

        if len(sequence) < 2 or len(places) < 2:
            continue  # skip sequences too short to mutate

        indices_to_mutate = random.sample(places, 2)
        sequence_list = list(sequence)

        for i in indices_to_mutate:
            original = sequence_list[i]
            options = [aa for aa in amino_acids if aa != original]
            new_aa = random.choice(options)
            sequence_list[i] = new_aa

        mutated_sequence = "".join(sequence_list)

        # Create a modified row
        new_row = row.copy()
        new_row["NES sequence"] = mutated_sequence
        new_row["label"] = 0  # mark as negative

        negative_rows.append(new_row)

    # Create a new DataFrame from the mutated rows
    neg_df = pd.DataFrame(negative_rows)

    # Define output path if not provided
    if output_csv_path is None:
        output_csv_path = nes_csv_path.replace(".csv", "_negatives.csv")

    neg_df.to_csv(output_csv_path, index=False)
    print(f"[✓] Negative dataset saved to {output_csv_path}")
    return output_csv_path


def main():
    # url = "http://prodata.swmed.edu/nes_pattern_location/"
    # fetch_nes_table_to_csv(url)
    generate_negative_csv_from_nes("nes_output.csv")

if __name__ == "__main__":
    main()
