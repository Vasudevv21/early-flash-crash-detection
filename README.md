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
