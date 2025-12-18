"""Explore the nvidia/Nemotron-Personas-USA dataset."""

from datasets import load_dataset
import json

def main():
    print("Loading nvidia/Nemotron-Personas-USA dataset...")
    ds = load_dataset("nvidia/Nemotron-Personas-USA", split="train")
    
    print(f"\nDataset size: {len(ds)} personas")
    print(f"\nColumns/Fields: {ds.column_names}")
    
    # Save first 4 personas to JSON
    personas = [ds[i] for i in range(4)]
    
    output = {
        "dataset_info": {
            "name": "nvidia/Nemotron-Personas-USA",
            "size": len(ds),
            "columns": ds.column_names
        },
        "sample_personas": personas
    }
    
    with open("sample_personas.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print("\nSaved first 4 personas to sample_personas.json")

if __name__ == "__main__":
    main()
