"""
Stratified Validation Split & Evaluation Framework for Virtual Cell Challenge.

This module splits available perturbations/cells into Train and Validation subsets
to evaluate local model improvements before submitting to the leaderboard.
"""

import os
import json
import numpy as np
import pandas as pd
import anndata as ad
from typing import Tuple, List

def create_perturbation_split(
    pert_csv: str = "data/pert_counts.csv",
    val_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[List[str], List[str]]:
    """
    Creates an unseen-perturbation split (Holdout Zero-Shot evaluation).
    """
    df = pd.read_csv(pert_csv)
    all_perts = df["target_gene"].tolist()
    
    np.random.seed(seed)
    shuffled = np.random.permutation(all_perts)
    
    n_val = int(len(shuffled) * val_ratio)
    val_perts = sorted(shuffled[:n_val].tolist())
    train_perts = sorted(shuffled[n_val:].tolist())
    
    print(f"Total Perturbations: {len(all_perts)}")
    print(f"Training Perturbations: {len(train_perts)}")
    print(f"Validation (Holdout) Perturbations: {len(val_perts)}")
    
    split_dir = "data/splits"
    os.makedirs(split_dir, exist_ok=True)
    
    pd.DataFrame({"target_gene": train_perts}).to_csv(os.path.join(split_dir, "train_perts.csv"), index=False)
    pd.DataFrame({"target_gene": val_perts}).to_csv(os.path.join(split_dir, "val_perts.csv"), index=False)
    
    return train_perts, val_perts

if __name__ == "__main__":
    create_perturbation_split()
