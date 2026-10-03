"""
Baseline Model: Control Mean & Stratified Sampling Baseline for Virtual Cell Challenge 2026.

Strategy:
- For each context (A, B, C):
  - In single-cell perturbation modeling, cells perturbed with gene X typically retain
    most of the background state of the context with specific DE shifts.
  - As the standard baseline (control-mean / sampled control), we take the context's
    non-targeting control cells and generate predictions for each of the 300 target perturbations
    (400 cells per perturbation per context).
  - Also include non-targeting controls (400 cells) so the dataset contains the reference control
    expected by differential expression / evaluation tools.
"""

import os
import json
import numpy as np
import pandas as pd
import anndata as ad
from scipy.sparse import csr_matrix, vstack

def build_baseline_prediction():
    data_dir = "data"
    manifest_path = os.path.join(data_dir, "manifest.json")
    gene_path = os.path.join(data_dir, "gene_names.csv")
    pert_path = os.path.join(data_dir, "pert_counts.csv")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    genes_df = pd.read_csv(gene_path)
    gene_names = genes_df["gene_name"].values
    perts_df = pd.read_csv(pert_path)
    target_perts = perts_df["target_gene"].tolist()
    cells_per_pert = manifest.get("cells_per_pert", 400)
    contexts = manifest.get("contexts", ["A", "B", "C"])

    print(f"Loaded manifest: {len(contexts)} contexts, {len(target_perts)} target perturbations, {cells_per_pert} cells/pert.")
    
    # We will build predictions context by context and concatenate them
    all_adatas = []

    np.random.seed(42)

    for ctx in contexts:
        ctx_file = os.path.join(data_dir, f"context_{ctx}.h5ad")
        print(f"\nProcessing Context {ctx} from {ctx_file}...")
        ctrl_adata = ad.read_h5ad(ctx_file)
        n_ctrl_cells = ctrl_adata.n_obs
        
        # We need predictions for all 300 perturbations (400 cells each)
        # plus the control cells for evaluation/reference
        all_perts = target_perts + ["non-targeting"]
        total_cells_ctx = len(all_perts) * cells_per_pert
        print(f"Context {ctx}: Generating {total_cells_ctx} cells ({len(all_perts)} conditions x {cells_per_pert} cells)...")

        # For single-cell data, sampling real control cells preserves realistic sparsity,
        # variance, and single-cell expression distribution across genes
        sampled_indices = []
        obs_perts = []
        obs_contexts = []

        for pert in all_perts:
            # Sample 400 cells with replacement from controls of this context
            idx = np.random.choice(n_ctrl_cells, size=cells_per_pert, replace=True)
            sampled_indices.extend(idx)
            obs_perts.extend([pert] * cells_per_pert)
            obs_contexts.extend([ctx] * cells_per_pert)

        # Slice sampled cells from ctrl_adata.X
        sampled_X = ctrl_adata.X[sampled_indices]

        obs_df = pd.DataFrame({
            "target_gene": obs_perts,
            "context": obs_contexts
        }, index=[f"{ctx}_cell_{i:06d}" for i in range(len(obs_perts))])

        ctx_pred_adata = ad.AnnData(
            X=sampled_X,
            obs=obs_df,
            var=pd.DataFrame(index=gene_names)
        )
        all_adatas.append(ctx_pred_adata)

    print("\nConcatenating all contexts...")
    full_prediction = ad.concat(all_adatas)
    print(f"Full Prediction shape: {full_prediction.shape}")
    print(f"Obs value counts per context:\n{full_prediction.obs['context'].value_counts()}")
    print(f"Unique perturbations: {full_prediction.obs['target_gene'].nunique()}")

    out_h5ad = "submissions/baseline_control_mean.h5ad"
    print(f"Saving to {out_h5ad}...")
    full_prediction.write_h5ad(out_h5ad)
    print("Done! Baseline prediction saved successfully.")

if __name__ == "__main__":
    build_baseline_prediction()
