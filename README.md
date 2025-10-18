# Amazon ML Challenge 2025 Toolkit

## Overview
This repository is a complete working environment for the Amazon Machine Learning Challenge 2025 — Smart Product Pricing track. It bundles the official problem brief, curated datasets, progressively richer baselines, high-throughput training scripts, documentation templates, and archived submissions so that teams can reproduce experiments end-to-end.

## Quick Links
- Workspace primer: `student_resource/README.md`
- Dataset reference: `student_resource/dataset/README.md`
- Pipeline catalog: `student_resource/src/README.md`
- Documentation index: `student_resource/Documentation/README.md`
- Submission logbook: `student_resource/Submissions/README.md`

## Challenge Background
- **Goal:** Predict optimal prices for retail catalog items using product descriptions and optional imagery.
- **Training data:** 75k labelled rows with `sample_id`, `catalog_content`, `image_link`, and `price`.
- **Evaluation:** Symmetric Mean Absolute Percentage Error (SMAPE); lower scores are better.
- **Constraints:** Predictions must be positive floats; external price lookup is forbidden (competition integrity rule).
- **Submission artefacts:** Upload `test_out.csv` for scoring and a one-page methodology report derived from `Documentation_template.md`.

## Repository Layout
- `student_resource/README.md` workspace guide covering quick-start instructions, experiment workflow, and the full challenge description.
- `student_resource/dataset/` canonical CSVs (`train.csv`, `test.csv`, samples) plus generated predictions and cached artefacts. See `student_resource/dataset/README.md` for schema details and workflow tips.
- `student_resource/src/` run-ready pipelines (`sample_code.py`, `spp_1.py … spp_10.py`, `clean.py`), utilities, and notebooks. Documented in `student_resource/src/README.md`.
- `student_resource/Documentation/` architecture diagrams, milestone reports, and narrative artefacts tracked in `student_resource/Documentation/README.md`.
- `student_resource/Submissions/` zipped leaderboard submissions, indexed by date and attempt. Managed via `student_resource/Submissions/README.md`.
- `student_resource/Documentation_template.md` organiser-provided one-page report scaffold.
- `requirements.txt` dependency pinning for core Python packages (NumPy, Pandas, SciPy, scikit-learn, XGBoost, LightGBM, etc.).
- `.gitignore` housekeeping file ignoring caches, large outputs, and temporary assets.
- `ml_.gif`, `68e8d1d70b66d_student_resource.zip` and similar artefacts retained for reference; not required for execution.

## Getting Started
1. **Clone / extract** this repository alongside the original dataset (if starting from the competition bundle).
2. **Create a virtual environment** (Python 3.10+ recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. **Verify data files** exist under `student_resource/dataset/` (`train.csv`, `test.csv`, `sample_test.csv`, `sample_test_out.csv`).
5. **Run a baseline** from the repository root:
   ```bash
   python student_resource/src/sample_code.py
   ```
   This produces `student_resource/dataset/test_out.csv`, confirming environment integrity.

## Environment & Hardware
- **Python:** 3.10 or newer; all scripts tested on CPython.
- **Operating systems:** Linux, macOS, and Windows (WSL) supported; ensure UTF-8 locale.
- **Hardware:** Minimum 16 GB RAM recommended for full TF-IDF pipelines; GPU optional (XGBoost auto-detects when available).
- **Parallelism:** Configure via environment variables (`N_JOBS`, `BLAS_THREADS`) to fit workstation capacity.
- **Networking:** Image downloads require outbound HTTP access; throttling mitigated through retries in `utils.py`.

## Modelling Pipelines
- **Baseline (`sample_code.py`)**: Toy predictor wiring the submission contract; useful for validating I/O expectations.
- **Sequential prototypes (`spp_1.py` … `spp_10.py`)**: Step-wise improvements covering feature engineering, target encoding, ensemble stacking, and environment toggles (e.g., `SPEED_MODE`, `USE_XGB`).
- **Production runner (`clean.py`)**: High-performance script with multiple presets (`blitz`, `fast`, `balanced`, `ultra`), optional GPU XGBoost, hashing vectorisers, and caching. Configure through environment variables; see inline docs.

All pipelines read data from `student_resource/dataset/` and write predictions or logs back into the same directory. Keep the dataset folder organised by periodically pruning obsolete `test_out_*.csv` files or moving heavy artefacts to a `cache/` subdirectory.

## Documentation & Reporting
- Architecture diagrams (`Documentation/arc*.png`) and milestone reports (`Documentation/report_*.pdf`) outline solution evolution, validation results, and governance notes.
- Submission archives (`Submissions/datavengers_<date>_<run>.zip`) pair predictions with supporting evidence. A tracking table in `student_resource/Submissions/README.md` links each archive to its documentation.
- Use `Documentation_template.md` to generate the one-page summary required by the competition organisers. Reference the appropriate diagram/report for every final submission.

## Dataset Summary
- Shared schema across `train.csv` and `test.csv`; only `train.csv` contains `price`.
- `catalog_content` combines title, bullet points, pack size, and marketing copy; expect irregular punctuation and mixed casing.
- `image_link` frequently points to Amazon CDN assets; `utils.py` handles bulk downloads with caching.
- Additional derived files (`test_out*.csv`, `cv_metrics_log.csv`, `final_model.pkl`) accumulate in `student_resource/dataset/`; manage via `.gitignore`.

## Workflow Recommendations
1. **Explore** dataset characteristics via `student_resource/dataset/README.md` and notebook experiments (`src/example.ipynb`).
2. **Prototype** with `src/spp_*.py`, iterating on feature sets and model choices. Capture cross-validation metrics in `dataset/cv_metrics_log.csv`.
3. **Scale** using `src/clean.py`, tuning environment variables for hardware constraints.
4. **Package** predictions (`test_out.csv`) with contextual notes (model config, commit SHA) and store them under `student_resource/Submissions/`.
5. **Document** every major change in the `Documentation/` folder; ensure the README indices stay up to date.

## Testing & Validation
- Use cross-validation splits embedded in `spp_1.py`/`clean.py` to measure SMAPE, RMSE, MAE, and R².
- Keep `dataset/cv_metrics_log.csv` under version control only if it adds insight; otherwise archive per experiment.
- When introducing new features, run regression checks by comparing SMAPE deltas against previous benchmarks.
- For lightweight sanity checks, craft unit tests or assertions in notebooks that validate feature extraction logic.

## Reproducibility & Versioning
- Tag code changes with semantic descriptions (e.g., `feat/tfidf-svd-upgrade`) and capture commit SHA in submission notes.
- Record environment fingerprints (`pip freeze`) inside each submission archive for future rebuilds.
- Update `requirements.txt` whenever dependencies change; align environment upgrades with documentation updates.
- Maintain the submission table in `student_resource/Submissions/README.md` as the authoritative mapping from archive to code state.

## Dependency Snapshot
Key Python packages captured in `requirements.txt`:
- `numpy`, `pandas`, `scipy` — numerical and tabular foundations.
- `scikit-learn` — feature engineering, model selection, evaluation metrics.
- `xgboost`, `lightgbm` — gradient boosting implementations used across pipelines.
- `tqdm`, `requests`, `joblib`, `psutil`, `threadpoolctl` — utility libraries supporting progress visualisation, data acquisition, parallelism control, and environment introspection.
- `jupyter` — notebook interface for exploratory analysis.

## Housekeeping
- `.gitignore` excludes Python caches, virtual environments, transient notebook checkpoints, large generated outputs (`test_out*.csv`, feature caches, image downloads), and temporary submission staging.
- Keep sensitive information (API keys, credentials) out of the repository. If needed, use environment variables or secrets management tools.
- For large artefacts (models, zipped submissions), consider offloading storage to cloud buckets or Git LFS and reference locations in the relevant README sections.

## Submission Checklist
- Regenerate `student_resource/dataset/test_out.csv` with the selected pipeline.
- Confirm schema alignment with `sample_test_out.csv` (column names, row count, positive prices).
- Bundle predictions, notes, logs, and environment details into a dated zip inside `student_resource/Submissions/`.
- Update the submission index and associated documentation/report links.
- Review competition rules on data usage and licensing prior to uploading.

## Data Governance & Compliance
- **Data lineage:** All inputs originate from the organiser-supplied CSVs and optional images fetched via `image_link`. External augmentation is prohibited.
- **Licensing:** Ensure derived models comply with MIT/Apache 2.0 limits specified by the organisers. Document licences for third-party libraries in submission reports.
- **PII handling:** Neither datasets nor scripts should ingest personal data. If additional sources are introduced, capture justification and sanitisation steps.
- **Reproducibility logs:** For every leaderboard upload, store commit SHA, environment manifest, and parameter overrides in the associated submission archive.
- **Security:** Never embed credentials in code. Use environment variables for optional service keys (e.g., S3 credentials when running outside the competition sandbox).

## Feature Engineering Highlights
- **Textual features:** TF-IDF (word and character n-grams), SVD dimensionality reduction, hashing vectors for blitz mode, regex-derived pack sizes, and lexical statistics (length, digit ratios, punctuation frequency).
- **Structured signals:** Pack-of counts, unit normalisation (`g`, `ml`, `count`), value-per-pack ratios, log-transformed quantities, and flag columns for marketing keywords (organic, keto, etc.).
- **Image hooks:** `utils.py` enables parallel image downloads to support CNN-based augmentation; store processed features separately (e.g., `dataset/cache/image_embeddings.parquet`).
- **Target encodings & ensembles:** Advanced pipelines (`spp_7+`, `clean.py`) support CV-based target encoding and stacked regressors (SGD, PassiveAggressive, Ridge, XGBoost).
- Document new features in code comments and append brief descriptions to `student_resource/src/README.md` to help reviewers understand the evolving pipeline.

## Benchmarking & Metrics Tracking
- **Primary metric:** SMAPE on validation folds; report mean ± std across repeats.
- **Secondary metrics:** RMSE, MAE, and R² recorded automatically by `spp_1.py` and `clean.py`.
- **Logging:** Append each experiment’s results to `dataset/cv_metrics_log.csv` with timestamp, commit SHA, and configuration flags.
- **Dashboarding:** Optionally mirror results into spreadsheet/MLFlow dashboards—reference the external location in the submission README table.
- **Regression testing:** Maintain historical SMAPE thresholds; flag degradations >1% absolute for investigation before shipping a submission.

## Troubleshooting & Known Issues
- **Memory pressure during TF-IDF:** Reduce `WORD_TFIDF`, `CHAR_TFIDF`, or switch `SPEED_MODE` to `ultra`. Confirm swap usage before incremental tuning.
- **XGBoost GPU errors:** Set `XGB_USE_GPU=off` if the environment lacks CUDA-capable hardware or has driver conflicts.
- **Image download throttling:** Rerun `download_images` with a smaller pool size (`Pool(20)`) or add exponential backoff around failed URLs.
- **UnicodeDecodeError on CSV load:** Ensure the interpreter uses UTF-8 (`export PYTHONUTF8=1`) and avoid editing CSVs in tools that change encoding.
- **Inconsistent leaderboard score:** Verify submission ordering by `sample_id` and confirm no rounding/formatting differences relative to `sample_test_out.csv`.

## FAQ
- **Q:** *Can we use transformer models?*  
  **A:** Yes, provided the final model stays within the 8B parameter limit and complies with licensing. Document pretrained sources in the submission report.
- **Q:** *Where should large model checkpoints live?*  
  **A:** Store outside the repo (cloud bucket, artifact store). Log the access path in `Documentation/report_*.pdf` and reference it in the submissions table.
- **Q:** *How do we add a new pipeline?*  
  **A:** Place the script in `student_resource/src/`, update `student_resource/src/README.md` with usage instructions, and include dependency changes in `requirements.txt`.
- **Q:** *What if the dataset schema changes?*  
  **A:** Update `student_resource/dataset/README.md`, adjust parsing logic, and regenerate baseline outputs before pushing changes. Note the delta in documentation.
- **Q:** *Who approves submissions?*  
  **A:** Follow your team’s governance—typically the assigned lead reviews `test_out.csv`, cross-validates SMAPE, and signs off in the submission README table.

## Roadmap & Future Enhancements
- Integrate transformer-based text embeddings with efficient fine-tuning strategies (LoRA/PEFT) while respecting parameter budgets.
- Add automated evaluation notebooks that compare SMAPE trends and generate executive summaries.
- Extend `clean.py` to save inference-ready pipelines (`.pkl` or `onnx`) for deployment experiments.
- Investigate multimodal fusion by combining image embeddings with textual features via late fusion ensemble.
- Implement continuous integration hooks that lint scripts, check formatting, and run lightweight unit tests on feature extraction utilities.

## Support & Contribution
- Pull requests should update the relevant README tables when new scripts, reports, or submissions are added.
- When introducing new pipelines or utilities, document expected inputs/outputs, configurable parameters, and resource requirements in `student_resource/src/README.md`.
- Coordinate dependency changes via `requirements.txt` and test compatibility across major OS/hardware configurations.
- For questions or onboarding, consult the nested READMEs first; escalate to project maintainers with concrete context (affected script, dataset slice, expected behaviour).

For deeper context on datasets, scripts, documentation, or submission governance, consult the nested READMEs inside `student_resource/`. Each directory maintains its own detailed guide aligned with the overall workflow described here.
