# Code availability

The PnAL implementation and supplementary evaluation runner are available in
`src/pnal/pnal_algorithm.py`. The released manuscript-level results are
organized under `data/paper/`, with testbed protocols and source-run mapping
documented in `docs/experiment_protocol.md`.

The program loads author-generated checkpoints, semantic calibration files,
and saved evaluation records from the configured run directories. Those trained
checkpoint artifacts remain with the original research runs. The publication
tables can be inspected independently from those artifacts.

The source contains the calibrated semantic gate, neuro-fuzzy risk screening,
SafetyShield, conflict coordination, model definitions, and evaluation-time
ablation definitions. Optional offline semantic-template services read
credentials from environment variables.
