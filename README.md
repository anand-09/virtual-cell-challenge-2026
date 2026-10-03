# Virtual Cell Challenge 2026

Zero-shot perturbation prediction and multi-context transcriptomic response modeling.

## Project Structure
- `src/baseline_model.py`: Control-mean and stratified sampling baseline generator
- `src/validation_split.py`: Train/Validation holdout split for zero-shot testing
- `models/perturbation_mlp.py`: Deep MLP perturbation response network
- `data/`: Dataset manifests, perturbation lists, and gene axis references
- `submissions/`: Output `.vcc` packages for the leaderboard

## Evaluation Metrics (Arc Institute VCC Profile)
- `mae` (Mean Absolute Error)
- `discrimination_score_l1`
- `overlap_at_N` (Top differentially expressed genes overlap)

## Author
- Anand Singh (`as7376835@gmail.com`)
