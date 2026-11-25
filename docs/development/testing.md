# Testing

```bash
pytest tests/                      # full suite (~2-3 min on a single GPU)
pytest tests/test_configs.py       # L1: every YAML parses and builds
pytest tests/test_forward_shapes.py # L2: every model forward-passes correctly
pytest tests/test_datasets.py      # data loaders: splits, labels, no train/test leakage
```
