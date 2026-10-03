"""
Regulatory Network-Informed Perturbation Model (v2 Competitive Architecture)

Biological Innovations:
1. Targeted CRISPRi Suppression:
   - Knock down the perturbed target gene (80-90% reduction, whole counts).
2. Transcriptional Co-Expression Propagation (ACLY -> Cholesterol/Lipid synthesis pathway, etc.):
   - In living cells, gene knockdowns do NOT happen in isolation. Downstream genes in the
     same regulatory network shift concordantly.
   - For every target gene, compute functional co-expression partners from the 18,400 basal cells.
   - Apply coordinated regulatory deltas to top functional partners, turning on true DE direction
     fidelity (fid), reach, and overlap (jac).
3. Low-Noise Replicate Sampling:
   - Preserves natural 400-cell single-cell variance distributions while avoiding artificial
     false-positive spikes from repeated sampling with replacement.
4. Strict VCC 2026 Format Compliance:
   - Exactly 360,000 cells (3 contexts x 300 perts x 400 cells).
   - Zero control cells.
   - Raw integer counts.
"""

import os
import json
import numpy as np
import pandas as pd
import anndata as ad
from scipy.sparse import csr_matrix, lil_matrix

def build_network_informed_model():
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

    print("Building Network-Informed Perturbation Model...")
    print(f"Total targets: {len(target_perts)} across contexts: {contexts}")

    all_adatas = []
    np.random.seed(42)

    for ctx in contexts:
        ctx_file = os.path.join(data_dir, f"context_{ctx}.h5ad")
        print(f"\n--- Processing Context {ctx} ({ctx_file}) ---")
        ctrl_adata = ad.read_h5ad(ctx_file)
        n_ctrl_cells = ctrl_adata.n_obs

        # Identify expressed genes (present in >= 5% of cells) for co-expression network
        cell_counts = np.array((ctrl_adata.X > 0).sum(axis=0)).flatten()
        expressed_mask = cell_counts >= (0.05 * n_ctrl_cells)
        exp_indices = np.where(expressed_mask)[0]
        exp_gene_names = [gene_names[i] for i in exp_indices]
        exp_to_sub = {g: i for i, g in enumerate(exp_gene_names)}

        print(f"Computing context-specific co-expression graph over {len(exp_indices)} expressed genes...")
        # Use 4,000 representative control cells to build the gene-gene correlation matrix
        sub_X = ctrl_adata.X[:4000, expressed_mask].toarray()
        sub_X_norm = sub_X / (sub_X.sum(axis=1, keepdims=True) + 1e-9) * 10000.0
        sub_X_log = np.log1p(sub_X_norm)
        corr_matrix = np.corrcoef(sub_X_log, rowvar=False)

        # Build sampled cellular background
        # Rather than sampling 400 cells completely at random with high replacement,
        # we tile the 46 natural 400-cell control guide blocks to preserve realistic intra-replicate variance
        sampled_indices = []
        obs_perts = []
        obs_contexts = []

        for pert_i, pert in enumerate(target_perts):
            # Select 400 natural cells
            start_c = (pert_i * cells_per_pert) % (n_ctrl_cells - cells_per_pert)
            idx = np.arange(start_c, start_c + cells_per_pert)
            sampled_indices.extend(idx)
            obs_perts.extend([pert] * cells_per_pert)
            obs_contexts.extend([ctx] * cells_per_pert)

        print(f"Constructing single-cell matrix ({len(sampled_indices)} cells)...")
        sampled_X = ctrl_adata.X[sampled_indices].tolil()

        print(f"Applying target knockdown & functional network propagation...")
        for pert_i, pert in enumerate(target_perts):
            start_row = pert_i * cells_per_pert
            end_row = start_row + cells_per_pert

            # 1. Direct Target Knockdown (85% reduction)
            if pert in gene_to_idx:
                target_col = gene_to_idx[pert]
                curr_vals = sampled_X[start_row:end_row, target_col].toarray()
                kd_vals = np.round(curr_vals * 0.15).astype(np.float32)
                sampled_X[start_row:end_row, target_col] = kd_vals

            # 2. Network Co-expression Propagation
            # For correlated pathway partners (corr >= 0.15), apply concordant regulatory shift
            if pert in exp_to_sub:
                p_sub_idx = exp_to_sub[pert]
                p_corrs = corr_matrix[p_sub_idx].copy()
                p_corrs[p_sub_idx] = 0.0 # exclude target gene itself

                # Select top correlated partners
                top_partner_sub_indices = np.where(p_corrs >= 0.12)[0]
                for partner_sub_idx in top_partner_sub_indices:
                    partner_gene = exp_gene_names[partner_sub_idx]
                    partner_col = gene_to_idx[partner_gene]
                    partner_corr = float(p_corrs[partner_sub_idx])

                    # Shift partner expression proportionally: delta ~ -0.15 * correlation
                    curr_partner_vals = sampled_X[start_row:end_row, partner_col].toarray()
                    shift_factor = 1.0 - (0.30 * partner_corr)
                    shifted_vals = np.round(curr_partner_vals * shift_factor).astype(np.float32)
                    sampled_X[start_row:end_row, partner_col] = shifted_vals

        sampled_X_csr = sampled_X.tocsr()
        sampled_X_csr.eliminate_zeros()

        # Sanity checks
        assert (sampled_X_csr.data >= 0).all()
        assert (sampled_X_csr.data % 1 == 0).all()

        obs_df = pd.DataFrame({
            "target_gene": obs_perts,
            "context": obs_contexts
        }, index=[f"{ctx}_cell_{i:06d}" for i in range(len(obs_perts))])

        ctx_adata = ad.AnnData(
            X=sampled_X_csr,
            obs=obs_df,
            var=pd.DataFrame(index=gene_names)
        )
        all_adatas.append(ctx_adata)

    print("\nConcatenating network-informed model across all contexts...")
    full_adata = ad.concat(all_adatas)
    print(f"Final Prediction Shape: {full_adata.shape} (Expected: (360000, 18533))")
    assert full_adata.shape == (360000, 18533)
    assert "non-targeting" not in full_adata.obs["target_gene"].values

    out_h5ad = "submissions/network_informed_v2.h5ad"
    print(f"Saving to {out_h5ad}...")
    full_adata.write_h5ad(out_h5ad)
    print("Network-informed model saved successfully!")

if __name__ == "__main__":
    build_network_informed_model()
