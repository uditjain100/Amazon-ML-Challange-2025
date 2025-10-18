# Source Workspace

The `src/` directory contains the run-ready scripts, reusable helpers, and exploratory notebooks that power the Smart Product Pricing solution. Everything that executes training, validation, inference, or auxiliary data processing lives here.

## Directory Map
- `clean.py` production-grade training runner with multiple speed presets, caching, and ensembling hooks.
- `spp_1.py` … `spp_10.py` sequential prototyping pipelines illustrating diverse feature engineering and model configurations.
- `sample_code.py` minimal baseline demonstrating the submission contract (`test_out.csv` creation).
- `utils.py` shared utilities (currently focused on image downloading).
- `example.ipynb` scratch notebook for ad-hoc experiments or visualisation.
- `__pycache__/` Python bytecode cache generated automatically; safe to ignore but leave out of version control.

> Tip: keep any additional notebooks or helper modules inside this folder so relative imports (e.g., `from src.utils import download_images`) continue to work.

## Pipeline Overview

| Script           | Purpose                                                | Typical Usage                                                   | Key Outputs                                    |
|------------------|--------------------------------------------------------|------------------------------------------------------------------|------------------------------------------------|
| `sample_code.py` | Sanity-check baseline producing random predictions     | `python src/sample_code.py`                                      | `dataset/test_out.csv`                         |
| `spp_1.py`       | TF-IDF + SVD + boosted trees with structured features | `python src/spp_1.py`                                            | `dataset/test_out.csv`, `cv_metrics_log.csv`   |
| `spp_2.py`–`spp_9.py` | Iterative enhancements (target encoding, ensembles, etc.) | `python src/spp_k.py`                                            | varies (predictions, logs, cache files)        |
| `spp_10.py`      | Hybrid online learners + optional XGBoost, tuned via env vars | `SPEED_MODE=ultra USE_XGB=0 python src/spp_10.py`               | `dataset/test_out.csv`, diagnostics            |
| `clean.py`       | Comprehensive runner supporting TF-IDF, hashing, CV stacking, GPU XGB | `SPEED_MODE=fast USE_SVD=1 python src/clean.py`           | `test_out.csv`, `cv_metrics_log.csv`, caches   |

Use the `SPEED_MODE`, `USE_SVD`, `USE_XGB`, and similar environment variables (documented inline) to customise runtime behaviour without editing the scripts.

## Image Download Utilities (`utils.py`)

`utils.py` exposes two high-level helper functions:

```python
from src.utils import download_image, download_images

# Download a single image
download_image("https://m.media-amazon.com/images/I/71XfHPR36-L.jpg", "dataset/images")

# Download many images in parallel
links = df["image_link"].dropna()
download_images(links, "dataset/images")
```

- `download_image(image_link, savefolder)` fetches a single image and writes it to `savefolder`, skipping files that already exist.
- `download_images(image_links, download_folder)` creates the target directory (if needed) and dispatches downloads across a multiprocessing pool (default pool size 100) while displaying a `tqdm` progress bar.

### Tips
- Install `tqdm` to preserve the progress bar; without it the loop still runs but without feedback.
- Reduce the pool size (e.g., by editing the `Pool(100)` constant or monkeypatching) when running on memory-constrained machines or when servers throttle parallel requests.
- Handle intermittent network failures by re-running `download_images`—existing files are skipped so only missing images are retried.
- Store downloaded images under `dataset/images/` (or similar) so that pipelines can easily load file paths alongside the tabular data.
- Consider writing a small manifest (`images_manifest.csv`) that maps `sample_id` to the downloaded filename; this enables synchronous joins between text and image-based models.

## Running Scripts

- **Baseline check:** `python src/sample_code.py`
- **Prototype pipeline:** `python src/spp_1.py`
- **Fast training:** `SPEED_MODE=fast python src/clean.py`
- **Lightweight inference:** `SPEED_MODE=blitz USE_SVD=0 python src/clean.py`

All scripts assume the working directory is the repository root (`student_resource/src` is imported as a package). If you run from notebooks or alternate shells, ensure `sys.path` includes the project root.

## Configuration Reference

Most pipelines expose knobs via environment variables:

- `SPEED_MODE` (`blitz`, `fast`, `balanced`, `ultra`): adjusts TF-IDF size, SVD dimensions, and learner choice.
- `USE_SVD`, `USE_CHAR`, `USE_XGB`, `USE_SGD`, `USE_RIDGE`: toggle components in `clean.py`/`spp_10.py`.
- `MAX_CHARS`, `WORD_TFIDF`, `CHAR_TFIDF`, `HASH_WORD`, `HASH_CHAR`: control text preprocessing limits.
- `N_JOBS`, `BLAS_THREADS`, `OOF_FOLDS`: manage parallelism and cross-validation intensity.

Consult the docstrings and inline comments inside each script for the complete list.

## Extending the Module

- Place new helpers in separate modules (e.g., `augmentations.py`) and import them where needed.
- Keep data paths relative to `dataset/` to maintain portability across environments.
- Document new environment variables or configuration options here so other contributors discover them easily.

## Notebook Usage

- Launch notebooks from the project root (`jupyter lab` or `jupyter notebook`).
- Within notebooks, add `sys.path.append("student_resource/src")` if you want to import modules by name (e.g., `import utils`).
- Use notebooks to prototype feature extraction, then upstream reusable code into `.py` modules for repeatability.

## Housekeeping

- Ignore `__pycache__/` and large intermediate artefacts by updating `.gitignore`.
- Keep external dependencies declared in the project-level documentation or `requirements.txt` (if present).
- When you add new scripts, update the pipeline table above with a one-line description and sample command.

This README should evolve with the codebase—refresh the tables and sections as pipelines mature or new utilities are added.
