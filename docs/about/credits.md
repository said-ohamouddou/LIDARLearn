# Credits

LIDARLearn builds on the outstanding work of the open-source point cloud community. Full per-model attribution — author, venue, original repository, and upstream licence — lives in [`THIRD_PARTY_NOTICES.md`](https://github.com/said-ohamouddou/LIDARLearn/blob/main/THIRD_PARTY_NOTICES.md); [`docs/point_cloud_methods.csv`](https://github.com/said-ohamouddou/LIDARLearn/blob/main/docs/point_cloud_methods.csv) has paper titles and venues.

BibTeX entries for **every** cited work (datasets, supervised backbones, SSL methods, PEFT strategies) are in **[`scripts/reports/references.bib`](https://github.com/said-ohamouddou/LIDARLearn/blob/main/scripts/reports/references.bib)**.

Every LaTeX report from `scripts/reports/*.py` cites from this file and **copies it into `<exp_dir>/latex/`**, so the generated `.tex` compiles on its own:

```bash
cd <exp_dir>/latex && pdflatex <report>.tex && bibtex <report> && pdflatex <report>.tex && pdflatex <report>.tex
```

Framework inspiration: [Pointcept](https://github.com/Pointcept/Pointcept), [OpenPoints](https://github.com/guochengqian/openpoints), [Torch-Points3D](https://github.com/torch-points3d/torch-points3d), [Learning3D](https://github.com/vinits5/learning3d).

KAN layers used by PointKAN and KAN-DGCNN are built on top of [efficient-kan](https://github.com/Blealtan/efficient-kan) (MIT) — a fast, drop-in replacement for the original Kolmogorov–Arnold Network implementation.
