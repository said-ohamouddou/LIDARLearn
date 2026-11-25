# Installation

LIDARLearn's full test suite (~2,200 tests) passes on two setups. Pick one and use its values in the steps below:

| | **A. Reference (paper results)** | **B. Newer PyTorch** |
|---|---|---|
| Python | 3.11 | 3.11 |
| PyTorch / torchvision | 2.4.1 / 0.19.1 | 2.8.0 / 0.23.0 |
| CUDA wheels (`<cu>`) | `cu118` | `cu128` |
| `nvcc` for the extensions | 11.x | 12.x |
| PyG wheel index | `torch-2.4.1+cu118` | `torch-2.8.0+cu128` |

Setup A reproduces the environment used for the paper. To reproduce the paper's code exactly as well, check out tag [`v0.5.0`](https://github.com/said-ohamouddou/LIDARLearn/tree/v0.5.0).

`torch-scatter` and `torch-cluster` are installed from the official PyG wheel index because their wheels are built for one specific (torch × CUDA) pair; a plain `pip install` will either fail or pull the wrong build.

## 1. Create a fresh conda environment

```bash
conda create -n lidarlearn python=3.11 -y
conda activate lidarlearn
```

## 2. Install PyTorch

```bash
# A. Reference
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu118
# B. Newer PyTorch
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
```

## 3. Install `torch-scatter` and `torch-cluster` from the PyG wheel index

Pass the torch+CUDA tag in the URL; this is what avoids the "wrong ABI / no CUDA" failures:

```bash
# A. Reference
pip install torch-scatter torch-cluster -f https://data.pyg.org/whl/torch-2.4.1+cu118.html
# B. Newer PyTorch
pip install torch-scatter torch-cluster -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
```

Verify they resolved to the CUDA wheels; a version ending in `+cpu` means you got the CPU-only build (CUDA wheels look like `1.6.3+pt28cu128`):

```bash
python -c "import torch_scatter, torch_cluster; print(torch_scatter.__version__, torch_cluster.__version__)"
```

## 4. Clone the repo and install the remaining requirements

```bash
git clone https://github.com/said-ohamouddou/LIDARLearn.git
cd LIDARLearn
pip install -r requirements.txt
```

`requirements.txt` only sets lower bounds for `torch`, `torchvision` and `numpy`, so it keeps the PyTorch you installed in step 2.

## 5. Build the CUDA C++ extensions

The extensions are compiled against your PyTorch, so `nvcc` must have the **same CUDA major version** as your torch build: 11.x for setup A, 12.x for setup B.

Check with `nvcc --version`. If your system toolkit doesn't match, install one into the conda env:

```bash
# Only needed when the system nvcc doesn't match (shown for setup B)
conda install -c conda-forge cuda-version=12.8 cuda-nvcc cuda-cudart-dev \
    libcublas-dev libcusparse-dev libcusolver-dev libcurand-dev
```

Then build:

```bash
bash extensions/install_extensions.sh
```

This builds 7 extensions: `pointnet2_ops`, `chamfer_dist`, `dela_cutils`, `pointops`, `ptv_modules`, `clip` and `index_max` (needed by SO-Net). The script:

- warns if `nvcc` and PyTorch disagree on the CUDA version;
- finds a conda-installed CUDA toolkit automatically;
- builds each extension independently and prints a pass/fail summary;
- exits with an error if any build failed.

Extensions you skip or fail to build don't break the rest of the library:

- `import models` still works.
- Models that need a missing extension are skipped, with one warning naming the missing package.
- Asking for one of those models raises an error that tells you what to install.

## 5b. (Optional) Install Mamba for PointMamba

PointMamba needs [`mamba-ssm`](https://github.com/state-spaces/mamba) and [`causal-conv1d`](https://github.com/Dao-AILab/causal-conv1d) (NVIDIA GPU only). Without them every other model works and PointMamba is skipped with a message. Install the prebuilt wheels that match your torch / CUDA / Python / C++ ABI (Python 3.11):

```bash
# A. Reference (torch 2.4.1 + cu118)
pip install https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.0.post8/causal_conv1d-1.5.0.post8+cu11torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
pip install https://github.com/state-spaces/mamba/releases/download/v2.2.4/mamba_ssm-2.2.4+cu11torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
# B. Newer PyTorch (torch 2.8.0 + cu128)
pip install https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
pip install https://github.com/state-spaces/mamba/releases/download/v2.2.5/mamba_ssm-2.2.5+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
# Both
pip install "transformers<5"
```

- Use `mamba-ssm` **2.2.x**: 2.3.x requires `triton>=3.5`, which conflicts with torch 2.8.
- `transformers<5` is needed because `mamba_ssm` imports names that transformers 5 removed.
- Other setups: pick the matching wheel from the [mamba](https://github.com/state-spaces/mamba/releases) and [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d/releases) release pages (check `python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"` for the `cxx11abi` part).

## 6. Verify the install

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_configs.py -q   # 1100+ config-build tests, ~1 min
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/ -q                  # full suite (~2,200 tests), ~2-3 min on GPU
```

Everything should pass. If a model's tests fail with `... is not in the models registry`, the error lists the missing extension or package; re-run step 5 and check that it built.

---

**Tested setups.** LIDARLearn has been developed and validated on:

- **OS** Ubuntu 22.04, NVIDIA driver 560
- **CPU** AMD Ryzen 7 5700X (8 cores / 16 threads)
- **GPU** NVIDIA GeForce RTX 4060 Ti (single-GPU; all runs validated on a single-GPU configuration)
- **Python** 3.11
- **Setup A:** PyTorch 2.4.1 (`+cu118`), torchvision 0.19.1, nvcc 11.8, torch-scatter / torch-cluster from the `torch-2.4.1+cu118` index; PointMamba with mamba-ssm 2.2.4 / causal-conv1d 1.5.0.post8
- **Setup B:** PyTorch 2.8.0 (`+cu128`), torchvision 0.23.0, nvcc 12.8, torch-scatter / torch-cluster from the `torch-2.8.0+cu128` index; PointMamba with mamba-ssm 2.2.5 / causal-conv1d 1.7.0
- **numpy** 1.26 and 2.x, **timm** 0.4.5, **ninja** (for C++/CUDA extension builds)

**Other PyTorch versions**

- PyTorch 2.4.1 is the minimum. Releases newer than 2.8 are likely to work but are not tested yet.
- For another torch/CUDA pair, change the versions in steps 2–3 and use a matching `nvcc` in step 5. The PyG index URL is `https://data.pyg.org/whl/torch-<torch version>+<cu tag>.html`.
- **Compatibility reports are very welcome**: please open an issue with your environment details.

**Multi-GPU / distributed**

- The code supports `torch.distributed` (see [`utils/dist_utils.py`](https://github.com/said-ohamouddou/LIDARLearn/blob/main/utils/dist_utils.py)), but every shipped benchmark ran on a single GPU.
- **Feedback from multi-GPU users is especially welcome**: please open an issue with your launcher, world size and any changes you needed.
