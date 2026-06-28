# PnAL: Confidence-Calibrated Action-Level Multi-Source Decision Fusion

This repository provides implementation code for **PnAL**, a confidence-calibrated action-level multi-source decision fusion framework for safety-critical multi-agent systems.

PnAL generates executable fused actions by combining:

- a learned decision source that proposes actions from local observations;
- an offline semantic-prior source that retrieves cached action-prior preferences;
- calibrated confidence gating for semantic-prior admission;
- a neuro-fuzzy risk source for risk-constrained filtering;
- priority-based local coordination for residual conflict arbitration;
- an auxiliary training-side adaptation interface for improving the learned decision source.

The implementation uses controlled multi-agent interaction testbeds as evaluation environments. These environments are provided as representative safety-critical interaction scenarios; the repository is organized around the reported algorithm rather than application-specific claims.

## Repository scope

This repository contains implementation code for the reported PnAL algorithm.

It does **not** provide additional experimental claims, supplementary experiments, additional figures, trajectory case studies, calibration-curve results, runtime-analysis claims, or extra robustness results beyond the manuscript.

The related manuscript has not been published yet. The repository therefore avoids journal, volume, issue, page, DOI, or formal article-citation claims.

## Repository structure

```text
PnAL-GitHub/
├── README.md
├── requirements.txt
├── pyproject.toml
├── LICENSE
├── CITATION.cff
├── SECURITY.md
├── .gitignore
├── configs/
│   └── example.env
├── docs/
│   └── code_availability.md
├── examples/
│   └── run_pnal.py
└── src/
    └── pnal/
        ├── __init__.py
        └── pnal_algorithm.py
```

## Installation

A Python environment with PyTorch and `highway-env` is required.

```bash
git clone https://github.com/RyanLiu1999/PnAL.git
cd PnAL
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Optional offline semantic-template generation

The semantic-prior component can use offline language assistance during template construction or update. Online execution should rely on cached templates, symbolic context matching, cached action-prior retrieval, and calibrated confidence checking.

API keys must be supplied through environment variables and must never be committed to the repository.

```bash
cp configs/example.env .env
# edit .env locally
export DEEPSEEK_API_KEY="your_api_key_here"
export DEEPSEEK_BASE_URL="https://ark.cn-beijing.volces.com/api/v3"
```

The code also supports fallback templates when no API key is provided.

## Running the implementation

```bash
python examples/run_pnal.py
```

The main algorithm implementation is located at:

```text
src/pnal/pnal_algorithm.py
```

## Citation

The related manuscript is currently unpublished. Until a formal paper citation is available, please cite this repository:

```bibtex
@misc{pnal_software_2026,
  title  = {PnAL: Implementation code for Confidence-Calibrated Action-Level Multi-Source Decision Fusion},
  author = {Liu, Ke and Ma, Jing and Lai, Edmund M-K},
  year   = {2026},
  note   = {Unpublished manuscript and software repository},
  url    = {https://github.com/RyanLiu1999/PnAL}
}
```

A formal article citation will be added after publication.

## Code availability statement

The repository provides implementation code for the reported PnAL algorithm. All experimental claims are based on the results reported in the manuscript.

## Security notice

Do not commit credentials, API keys, tokens, `.env` files, or local experiment outputs. Use environment variables for optional offline language-assisted template construction.
