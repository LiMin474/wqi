# Repository Guidelines

## Project Structure & Module Organization
Core Python code lives under `python_code/`. Use `python_code/common_codes/optimizers/` for individual optimizers, `python_code/common_codes/ensemble/` for stacking logic, `python_code/experiments/` for end-to-end experiment and figure scripts, and `python_code/analysis/` for post-run analysis such as divergence and statistical tests. Datasets are stored in `python_code/datasets/*.npz`, while generated artifacts belong in `python_code/results/`. The top-level `1_results/`, `2_results/`, and `3_results/` folders hold dataset-specific report outputs and figures.

## Build, Test, and Development Commands
Work from `python_code/` so imports resolve cleanly:

```powershell
cd python_code
$env:PYTHONPATH="."
python experiments/_run_unified_ensemble.py
python analysis/_analyze_divergence.py
python analysis/_statistical_tests.py
python experiments/_generate_paper_figures.py
```

`_run_unified_ensemble.py` generates the main JSON and CSV outputs. Run `_analyze_divergence.py` before `_statistical_tests.py`, because the latter depends on `results/divergence_analysis.json`. Use `_generate_paper_figures.py` only after results have been refreshed.

## Coding Style & Naming Conventions
Follow the existing Python style: 4-space indentation, snake_case for functions and variables, and short module-level helper functions for reusable calculations or file export logic. Keep new experiment scripts consistent with the current naming pattern, for example `_run_<dataset>_only.py` or `_verify_<scope>.py`. Prefer explicit paths built from `os.path.dirname(__file__)` instead of hard-coded absolute locations.

## Testing Guidelines
This repository uses script-based verification rather than a dedicated `pytest` suite. For changes to training or ensembling logic, rerun `experiments/_run_unified_ensemble.py` and at least one verification or analysis script such as `experiments/_verify_all.py` or `analysis/_statistical_tests.py`. Treat regenerated files in `python_code/results/` as part of the validation evidence and review diffs before committing.

## Commit & Pull Request Guidelines
Recent history uses short Chinese commit subjects with action-first wording such as `更新README`, `整理项目代码`, and `更新项目结构和完整实验结果`. Keep that style: start with a verb, name the affected area, and mention the outcome when useful. For pull requests, include a concise summary, list the scripts you reran, note any regenerated result files or figures, and attach screenshots only when visual outputs changed.

## Data & Output Hygiene
Do not overwrite source datasets casually. Keep large generated binaries such as `.docx`, `.png`, and `.pdf` updates intentional, and avoid committing scratch files, `__pycache__/`, or one-off debug outputs unless they are required for reproducibility.
