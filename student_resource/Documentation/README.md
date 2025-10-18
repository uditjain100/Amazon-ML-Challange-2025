# Documentation Artifacts

This folder aggregates visual summaries, analytical reports, and narrative artifacts that accompany modelling experiments for the Smart Product Pricing challenge. Treat this directory as the single source of truth for presentation-ready materials—everything here should be directly reusable when communicating progress to reviewers, stakeholders, or competition organisers.

## Contents
- `arc1.png` – `arc5.png`: Architecture diagrams capturing successive versions of the solution pipeline. Use these to illustrate design evolution, component interactions, and infrastructure hand-offs.
- `report_1.pdf` – `report_5.pdf`: Iterative write-ups covering data exploration, feature engineering choices, model configuration, validation results, and operational notes. Each report corresponds to a milestone submission or internal review cycle and should align with the filenames stored under `student_resource/Submissions/`.
- `README.md`: You are here—living documentation summarising what each artifact represents and how to extend the directory responsibly.

### Diagram Guidelines
- **arc1.png**: Baseline workflow mapping raw text ingestion to a TF-IDF + ridge regression pipeline. Highlights minimal infrastructure footprint.
- **arc2.png**: Introduces feature extraction branches (regex descriptors, statistics) feeding an ensemble stack.
- **arc3.png**: Adds image-based augmentation path leveraging downloaded product photos.
- **arc4.png**: Captures distributed training with GPU-accelerated XGBoost and caching layers.
- **arc5.png**: Final architecture blueprint with automated validation, submission packaging, and reporting hooks.

### Report Overview
- **report_1.pdf**: Exploratory data analysis (EDA), initial baselines, and preprocessing assumptions.
- **report_2.pdf**: Enhanced feature engineering, cross-validation setup, and interim leaderboard results.
- **report_3.pdf**: Ensemble strategies, error analysis, and ablation findings.
- **report_4.pdf**: Productionisation plan covering resource usage, inference latency, and monitoring.
- **report_5.pdf**: Final submission summary including model governance, licensing checks, and future work.

## Usage Notes
- Keep exported diagrams and reports here so they remain easy to locate when preparing competition submissions or stakeholder updates.
- Name new assets with incremental identifiers or ISO timestamps (e.g., `report_2025-10-14.pdf`, `arc_v7.png`) to avoid overwriting prior artifacts.
- Cross-reference these documents in `Documentation_template.md` when compiling the final one-pager required by the organisers. Summaries from each report should map directly to sections in that template.
- Store source files (draw.io diagrams, PowerPoint decks, notebook exports) alongside the finished outputs when possible; use subdirectories such as `Documentation/sources/` to keep editable versions organised.
- When publishing externally, ensure personally identifiable information or proprietary metrics are redacted before exporting to this folder.

## Maintaining the Archive
- **Versioning:** Increment filenames rather than replacing existing documents so that the evolution of the solution remains auditable. If a document supersedes a previous one, add a note in this README (e.g., “report_6.pdf replaces report_4.pdf for production recommendations”).
- **Indexing:** Update the tables above whenever a new diagram or report is added. Include a sentence summarising the intent of the artifact and the time period it covers.
- **Linkage to Submissions:** For every `datavengers_<date>_<run>.zip` in `student_resource/Submissions/`, either create or reference a matching report/diagram here. This ensures every leaderboard entry has accompanying documentation.
- **Backup:** Consider syncing this folder to a shared drive or document repository used by the team. Large files (>20 MB) might be better served in cloud storage with referenced links noted in this README.

## Suggested Workflow
1. Draft experiment notes in notebooks or markdown while running pipelines.
2. Export visualisations and tables, then compile them into a PDF following the reporting structure (problem context, methodology, experiments, results, next steps).
3. Update or create architecture diagrams reflecting any changes introduced by the experiment.
4. Save both diagram and report here, ensuring filenames include a run identifier or date.
5. Append a short blurb under the appropriate bullets above so collaborators know what changed.

## Tools & Templates
- Use the organiser-provided `Documentation_template.md` as a baseline for PDF report sections.
- Recommended diagram tools: draw.io, Excalidraw, Figma. Maintain the editable source file in `Documentation/sources/` to enable future updates.
- When converting notebooks to PDF, prefer `nbconvert` or `jupyter nbconvert --to webpdf` to preserve layout. Validate the export locally before committing.

Keeping this README fresh ensures new team members can quickly identify the latest authoritative documentation and trace the project narrative without digging through version history.
