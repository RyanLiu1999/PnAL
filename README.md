# PnAL: Confidence-Calibrated Action-Level Multi-Source Decision Fusion

Research implementation accompanying the manuscript **Confidence-Calibrated Action-Level Multi-Source Decision Fusion for Safety-Critical Multi-Agent Systems**.

The implementation includes learned action proposals, an offline semantic-template library, Platt-calibrated semantic admission, neuro-fuzzy state-risk assessment, action-dependent feasibility screening, priority-based multi-agent coordination, and PathNet/SAC training-side modules. The supplied entry point performs evaluation-time supplementary experiments using saved experimental runs.

## Layout

- `src/pnal/pnal_algorithm.py`: PnAL implementation and evaluation-time experiment runner.
- `data/paper/main_comparisons.csv`: Table 3 and Table 4 values from the corrected manuscript.
- `data/paper/controlled_pathways.csv`: matched IPPO and PnAL online-pathway comparisons.
- `data/paper/frozen_component_ablations.csv`: execution-time component comparisons.
- `data/paper/collision_mcnemar.csv`: reported paired collision tests.
- `data/paper/run_protocol_map.csv`: provenance map for the eight PnAL main settings.
- `docs/experiment_protocol.md`: configuration, evaluation procedure, and result provenance.
- `tools/check_paper_tables.py`: lightweight table and source integrity checks.

## Setup

Python 3.11+ is recommended. The reported supplementary environment used `highway-env==1.10.1` and `gymnasium==1.4.0`.

```bash
python -m pip install -r requirements.txt
python tools/check_paper_tables.py
```

### Evaluation with saved model artifacts

The supplied runner reads existing trained runs and their evaluation records; it does not retrain models. In Google Colab, mount Drive and place the original run directories under `/content/drive/MyDrive`. Choose `SHARD = "A"`, `"B"`, `"C"`, `"all"`, or `"summary"` at the beginning of `src/pnal/pnal_algorithm.py`, then run:

```bash
python src/pnal/pnal_algorithm.py
```

The selected model checkpoints are kept with the original experiment runs and are not included in this repository. The CSV files contain the supplied manuscript-level aggregate results. They are not presented as independently regenerated episode-level observations.

See [experiment protocol](docs/experiment_protocol.md) for the exact configuration map and analysis definitions.

## Citation

The related manuscript is under review. Cite this software repository until a published paper citation becomes available.
