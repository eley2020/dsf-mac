# Reproduction guide

1. Create a clean Python environment and install `requirements.txt` (a CUDA-enabled PyTorch build is required for GPU reproduction).
2. Download CHB-MIT from its official PhysioNet source and create the fold metadata using the project metadata-generation scripts.
3. Run the three risk-estimator scales with `scripts/train_chbmit_loocv.py`, using `--model progressive_internal_dsf`, `--progressive-internal-stage 3`, `--window-sec 5`, `--window-sec 7`, and `--window-sec 9` in separate runs. Use the values in `configs/chbmit_mainline_stage3.json`; paths must be changed to local paths.
4. Fuse the 5/7/9-second prediction streams with `fuse_multiscale_dsf_predictions.py` and select fusion on validation data only with `select_multiscale_fusion_weights.py`.
5. Fit and evaluate the causal controller with `evaluate_chbmit_learnable_alarm_controller.py`. The held-out test event is evaluated only after all validation choices are frozen.

The reported reference metrics and exact result roots are included in `results_manifest/`. This release is a reproduction aid, not a redistribution of the underlying datasets.
