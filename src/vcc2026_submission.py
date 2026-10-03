"""
VCC 2026 Official Specification-Compliant Competitive Prediction Generator

Strict Competition Invariants Enforced:
1. Shape: EXACTLY 360,000 cells (3 contexts x 300 perturbations x 400 cells).
2. NO non-targeting rows (The server rejects submissions containing non-targeting rows).
3. Raw counts in .X: Non-negative whole integers (no log1p, no normalize_total).
4. Sparse CSR matrix: Strict cap of at most 4,750,000,000 stored entries.
5. Max per-cell library size <= 1,000,000 counts.
6. Columns in .obs:
   - 'target_gene' (Gene symbols, e.g. ABCD1)
   - 'context' ('A', 'B', 'C')
7. Variables in .var: Exactly the 18,533 genes from gene_names.csv in identical order.
8. Biological Modeling:
   - Preserves true single-cell count distributions from each context's controls.
   - Applies CRISPRi knockdown suppression to the perturbed gene in whole-count space.
"""

import os
import json
import numpy as np
import pandas as pd
import anndata as ad
from scipy.sparse import csr_matrix, lil_matrix

def build_vcc2026_submission():
    data_dir = "data"
    gene_path = os.path.join(data_dir, "gene_names.csv")
    pert_path = os.path.join(data_dir, "pert_counts.csv")

    genes_df = pd.read_csv(gene_path)
    gene_names = genes_df["gene_name"].values
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    perts_df = pd.read_csv(pert_path)
    target_perts = perts_df["target_gene"].tolist()
    cells_per_pert = 400
    contexts = ["A", "B", "C"]

    print(f"Generating VCC 2026 official compliant prediction:")
    print(f"Contexts: {contexts}")
    print(f"Perturbations: {len(target_perts)}")
    print(f"Cells per perturbation: {cells_per_pert}")
    print(f"Total cells to produce: {len(contexts) * len(target_perts) * cells_per_pert:,} (EXACTLY 360,000)")

    all_adatas = []
    np.random.seed(42)

    for ctx in contexts:
        ctx_file = os.path.join(data_dir, f"context_{ctx}.h5ad")
        print(f"\nProcessing Context {ctx} from {ctx_file}...")
        ctrl_adata = ad.read_h5ad(ctx_file)
        n_ctrl_cells = ctrl_adata.n_obs

        # Sample exactly 400 cells for each of the 300 target perturbations (NO non-targeting)
        sampled_indices = []
        obs_perts = []
        obs_contexts = []

        for pert in target_perts:
            idx = np.random.choice(n_ctrl_cells, size=cells_per_pert, replace=True)
            sampled_indices.extend(idx)
            obs_perts.extend([pert] * cells_per_pert)
            obs_contexts.extend([ctx] * cells_per_pert)

        # Extract sampled cells as LIL matrix for editing
        sampled_X = ctrl_adata.X[sampled_indices].tolil()

        print(f"Applying CRISPRi knockdown suppression for {len(target_perts)} target genes in Context {ctx}...")
        for pert_i, pert in enumerate(target_perts):
            if pert in gene_to_idx:
                target_col = gene_to_idx[pert]
                start_row = pert_i * cells_per_pert
                end_row = start_row + cells_per_pert
                
                # Suppress target gene expression (knockdown ~90%), rounding to nearest whole integer
                current_vals = sampled_X[start_row:end_row, target_col].toarray()
                kd_vals = np.round(current_vals * 0.1).astype(np.float32)
                sampled_X[start_row:end_row, target_col] = kd_vals

        # Convert back to sparse CSR with whole integers
        sampled_X_csr = sampled_X.tocsr()
        sampled_X_csr.eliminate_zeros()

        # Verify values are whole non-negative integers
        assert (sampled_X_csr.data >= 0).all(), "Negative counts detected!"
        assert (sampled_X_csr.data % 1 == 0).all(), "Non-integer fractional counts detected!"

        obs_df = pd.DataFrame({
            "target_gene": obs_perts,
            "context": obs_contexts
        }, index=[f"{ctx}_cell_{i:06d}" for i in range(len(obs_perts))])

        ctx_pred = ad.AnnData(
            X=sampled_X_csr,
            obs=obs_df,
            var=pd.DataFrame(index=gene_names)
        )
        all_adatas.append(ctx_pred)

    print("\nConcatenating all contexts...")
    submission_adata = ad.concat(all_adatas)

    print("--- VERIFYING COMPETITION INVARIANTS ---")
    print(f"Shape: {submission_adata.shape} (Expected: (360000, 18533))")
    assert submission_adata.shape == (360000, 18533), f"Wrong shape: {submission_adata.shape}"
    assert "non-targeting" not in submission_adata.obs["target_gene"].values, "non-targeting rows present!"
    assert len(submission_adata.obs["target_gene"].unique()) == 300, "Wrong number of unique perts!"
    assert submission_adata.obs["context"].value_counts().to_dict() == {"A": 120000, "B": 120000, "C": 120000}
    
    # Check max count per cell
    cell_sums = np.array(submission_adata.X.sum(axis=1)).flatten()
    print(f"Max cell count: {cell_sums.max():,} (Cap: 1,000,000)")
    assert cell_sums.max() <= 1000000, "Cell count exceeds 1,000,000!"

    out_file = "submissions/vcc2026_submission.h5ad"
    print(f"\nWriting validated raw counts .h5ad to {out_file}...")
    submission_adata.write_h5ad(out_file)
    print("Done! Validated prediction ready for packaging.")

if __name__ == "__main__":
    build_vcc2026_submission()
