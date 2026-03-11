# Flash Crash Early Warning System

This repository provides a research-style Python architecture for flash-crash early warning with:

- Agent-based simulation of market dynamics and crash cascades
- Multi-stock training setup (NIFTY 50 + NIFTY Bank core names)
- Microstructure features (bid-ask spread, depth, volume dynamics)
- Cross-stock contagion graph via Graph Attention Network (GAT)
- Temporal modeling via Transformer encoder
- Pre-crash labeling for early-warning prediction

## Quick start

```bash
pip install yfinance pandas numpy torch torch-geometric scikit-learn
python flash_crash_system.py
```

By default the script runs with `use_simulator=True` to avoid API limits and quickly generate realistic training data.
Set `run_pipeline(use_simulator=False)` inside `flash_crash_system.py` to use Yahoo Finance data.

## Phase 1 upgrades now included

- Reproducibility via global seed setup (`numpy`, `torch`, Python random)
- Train/validation/test split
- Class-imbalance aware training with `BCEWithLogitsLoss(pos_weight=...)`
- Validation-based threshold tuning for best F1
- Checkpointing (`best_model.pt`) and run summary logging (`run_summary.json`)

Artifacts are saved in timestamped folders under `results/`.

## Yahoo intraday data note

Yahoo limits intraday history (for example `5m` is typically limited to ~60 days).
This project now automatically:

- adjusts `start` to the maximum supported lookback for intraday intervals, and
- falls back to `1d` data if intraday download returns empty (configurable).

You can control this via `Config` fields in `flash_crash_system.py`:
`auto_adjust_intraday_start`, `fallback_to_daily_on_failure`, and `fallback_interval`.


## Troubleshooting

If you see `ModuleNotFoundError: No module named "numpy"`, install dependencies in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Then run:

```bash
python flash_crash_system.py
```
