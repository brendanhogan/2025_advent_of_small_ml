"""
Step 1: Download the SYNTH dataset.
Just downloads - no processing.
"""

from datasets import load_dataset
import os

OUTPUT_DIR = "raw_synth"

def main():
    print("=" * 60)
    print("Downloading SYNTH dataset")
    print("=" * 60)
    
    # Download full dataset (cached to disk)
    print("\nDownloading from HuggingFace...")
    print("This will cache to hf_cache/ and save to raw_synth/")
    
    dataset = load_dataset(
        "PleIAs/SYNTH",
        split="train",
        data_files=["synth_*.parquet"],
        cache_dir="hf_cache",
        num_proc=8,  # Parallel download
    )
    
    print(f"\nDownloaded {len(dataset):,} samples")
    
    # Save to disk in arrow format (fast to load later)
    print(f"\nSaving to {OUTPUT_DIR}/...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dataset.save_to_disk(OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("Done!")
    print(f"  Total samples: {len(dataset):,}")
    print(f"  Saved to: {OUTPUT_DIR}/")
    print("\nNext: python 2_tokenize.py")
    print("=" * 60)


if __name__ == "__main__":
    main()

