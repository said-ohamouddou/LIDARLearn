# LIDARLearn

**A unified PyTorch library for 3D point cloud deep learning.**

LIDARLearn ships **59 ready-to-run model configurations**: 31 supervised backbones, 8 self-supervised (SSL) methods and 5 parameter-efficient fine-tuning (PEFT) strategies. All of them share one YAML config system, one command line and one reporting pipeline.

- **Tasks:** classification, few-shot classification, part segmentation, semantic segmentation, SSL pre-training.
- **Data:** ModelNet40, ScanObjectNN, ShapeNet-55, ShapeNet Part, S3DIS, and terrestrial laser scanning (STPCTLS tree species).
- **Benchmarking:** stratified *K*-fold cross-validation, Friedman/Nemenyi significance tests, and publication-ready LaTeX/CSV tables generated from your runs.

**Links:** [GitHub](https://github.com/said-ohamouddou/LIDARLearn) · [Paper (arXiv)](https://arxiv.org/abs/2604.10780) · [Colab demo](https://colab.research.google.com/drive/1TS8HDEjD5qwrzEmk3IZIDBmw29Zc5oAf)

## News

- **2026-11: PointMamba added.** LIDARLearn now supports [PointMamba](https://github.com/LMD0311/PointMamba) (NeurIPS 2024), a state-space (Mamba) backbone.
- **2026-11: Newer PyTorch and CUDA.** LIDARLearn now runs on PyTorch 2.8 with CUDA 12.8, in addition to PyTorch 2.4.1 with CUDA 11.8.

```{toctree}
:maxdepth: 2
:caption: Getting started
:hidden:

getting-started/installation
getting-started/pretrained
getting-started/quickstart
```

```{toctree}
:maxdepth: 2
:caption: Data and models
:hidden:

datasets
models
```

```{toctree}
:maxdepth: 2
:caption: User guide
:hidden:

guide/training
guide/augmentation
guide/visualization
guide/reports
```

```{toctree}
:maxdepth: 2
:caption: API
:hidden:

api/index
```

```{toctree}
:maxdepth: 2
:caption: Development
:hidden:

development/contributing
development/testing
```

```{toctree}
:maxdepth: 2
:caption: About
:hidden:

about/changelog
about/citation
about/license
about/credits
```

## Where to start

- New here? Follow [Installation](getting-started/installation.md), then the [Quick start](getting-started/quickstart.md).
- Choosing a model? See the [model catalogue](models.md).
- Preparing data? See [Datasets](datasets.md).
- Writing a paper? See [Automated reports](guide/reports.md).
