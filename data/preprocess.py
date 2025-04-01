# preprocess.py
import pandas as pd
import argparse

def preprocess(genomic_path, environmental_path, output_path):
    print("[INFO] Reading datasets...")
    genomic_df = pd.read_csv(genomic_path)
    env_df = pd.read_csv(environmental_path)

    print("[INFO] Initial shape of genomic data:", genomic_df.shape)
    print("[INFO] Initial shape of environmental data:", env_df.shape)

    print("[INFO] Dropping missing values...")
    genomic_df.dropna(inplace=True)
    env_df.dropna(inplace=True)

    print("[INFO] Merging datasets on 'location_id'...")
    merged_df = pd.merge(genomic_df, env_df, on='location_id', how='inner')

    print("[INFO] Final merged data shape:", merged_df.shape)
    print("[INFO] Saving to:", output_path)
    merged_df.to_csv(output_path, index=False)
    print("[SUCCESS] Preprocessing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess genomic and environmental data for model training.")
    parser.add_argument("--genomic", type=str, required=True, help="Path to the genomic dataset (CSV)")
    parser.add_argument("--environmental", type=str, required=True, help="Path to the environmental dataset (CSV)")
    parser.add_argument("--output", type=str, required=True, help="Path to save the cleaned/merged dataset")

    args = parser.parse_args()
    preprocess(args.genomic, args.environmental, args.output)

# python data/preprocess.py \
#   --genomic data/raw_data/genomic_data.csv \
#   --environmental data/raw_data/environmental_data.csv \
#   --output data/cleaned_data.csv