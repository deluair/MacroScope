#!/usr/bin/env python3

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data_generation.synthetic_data_generator import SyntheticDataGenerator

def main():
    print("Generating synthetic data...")
    generator = SyntheticDataGenerator()
    datasets = generator.generate_all_datasets()
    
    data_dir = project_root / "data" / "synthetic"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    for name, df in datasets.items():
        filepath = data_dir / f"{name}.csv"
        df.to_csv(filepath, index=False)
        print(f"Generated: {filepath}")
    
    print("Synthetic data generation completed successfully!")

if __name__ == "__main__":
    main()