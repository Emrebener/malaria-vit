from malaria_vit.config import RAW_DIR, SPLIT_CSV, SEED
from malaria_vit.data import write_split

if __name__ == "__main__":
    df = write_split(RAW_DIR, SPLIT_CSV, SEED)
    print("distinct patients:", df["patient_id"].nunique())
    print(df["split"].value_counts())
    print(df.groupby(["split", "label"]).size())
