# Contributing

Contributions are welcome! Whether you're adding a new backbone, a new dataset loader, a PEFT strategy, or fixing a bug, the workflow is:

1. **Fork** the repository and create a feature branch (`git checkout -b feat/my-model`).
2. **Add your model** in three steps — create `models/<name>/<name>.py` with a class inheriting `BasePointCloudModel` (or `BaseSegModel`) and decorated with `@MODELS.register_module()`, add its module to the `_MODEL_MODULES` list in `models/__init__.py`, and provide at least one YAML config under `cfgs/`.
3. **Run the test suite** — `pytest tests/` must pass. New models are automatically picked up by the L1 config-parse and L2 forward-shape tests.
4. **Follow the style** — keep YAML configs `_base_`-inherited, document non-obvious hyperparameters, and include the paper citation in your config header.
5. **Open a PR** describing the change, the reference paper/repository, and any benchmark numbers if available.

For bugs, feature requests, or questions, please open an issue on [GitHub](https://github.com/said-ohamouddou/LIDARLearn/issues) with a minimal reproducer (config + command + error log).

## Project structure

```
LIDARLearn/
├── cfgs/              # YAML configs (classification, segmentation, fewshot, pretrain)
├── datasets/          # Dataset loaders (ModelNet, ScanObjectNN, ShapeNet, S3DIS, STPCTLS)
│                      #   + augmentation.py (shared pointcloud transforms)
├── models/            # Backbones, SSL methods, PEFT adapters, seg wrappers
├── tools/             # Training runners (finetune, pretrain, seg, fewshot)
├── utils/             # Config, logging, metrics, checkpoint, parser
├── scripts/           # Training sweeps (train_*.sh), visualization
│   └── reports/       # LaTeX report generators (classification, partseg, semseg,
│                      #   fewshot, citations, Friedman significance) + shared references.bib
├── extensions/        # 7 CUDA C++ extensions + install_extensions.sh
├── tests/             # pytest suite (L1-L4) + smoke_train_*.sh
├── docs/              # Documentation, paper PDF, example report
└── main.py            # Unified entry point
```
