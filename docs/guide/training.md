# Training runners


Classification / fine-tuning goes through two wired-up runners, exposed in [`main.py`](https://github.com/said-ohamouddou/LIDARLearn/blob/main/main.py) as:

```python
from tools import finetune_run_net as finetune              # default
from tools.runner_finetune_test import run_net as finetune_test  # 3-way split
```

Select with the `--runner` flag (default `runner_finetune`):

| Runner | Data splits used | "Best checkpoint" chosen on | Final metric reported on | Use when |
|---|---|---|---|---|
| `runner_finetune` *(default)* | `dataset.train` + `dataset.val` | validation | **validation** | You have a train/val 2-way split (ModelNet40, STPCTLS CV folds, fewshot episodes). Validation is both the selection signal AND the reported number. |
| `runner_finetune_test` | `dataset.train` + `dataset.val` + `dataset.test` | validation | **test** | You have a genuine 3-way split and want the best checkpoint selected on val but the final headline number computed on a held-out test set (cleanest protocol for paper-style reporting). |

```bash
# 2-way (default): val metrics are the final numbers
python main.py --config cfgs/classification/DGCNN/STPCTLS/stpctls.yaml --mode finetune --exp_name dgcnn

# 3-way: best ckpt on val, final metrics on test
python main.py --config cfgs/classification/PointMAE/STPCTLS/stpctls_test.yaml \
    --mode finetune --runner runner_finetune_test --exp_name pointmae_test
```

Both runners write the same `cv_summary.csv` format, so the reports in `scripts/reports/` work with either.

**Single-run (non-CV) experiments also write a `cv_summary.csv`**, with each metric as `value ± 0.00`. Tables and Friedman/Nemenyi scripts therefore see the same format for 1 fold or K folds.
