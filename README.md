# DynoSys: A Dynamic Systems Framework for Multimodal Integration

This repository implements the **DynoSys framework**, a systems-level pipeline for integrating genetic, environmental, neurobiological, and behavioral data to model human development over time.

The framework supports:
- Longitudinal modeling (continuous outcomes)
- Time-to-event modeling (survival outcomes)
- Multimodal integration (PRS + environment + brain + behavior)
- Multi-task learning across phenotypes

---

## 📁 Core Scripts

| Step | Script | Description |
|------|------|-------------|
| 00 | `00_build_unified_tables.py` | Build unified longitudinal dataset |
| 01 | `01_benchmark_unified_models_generic.py` | Classical baseline models |
| 02 | `02_discover_domain_relationships_generic.py` | Domain-level relationship discovery |
| 03 | `03_build_forward_tables_generic.py` | Build forward prediction tables |
| 04 | `04_train_forward_benchmarks_generic.py` | Train forward prediction models |
| 05 | `05_train_discrete_time_hazard_generic.py` | Discrete-time survival models |
| 06 | `06_train_multimodal_neural_generic.py` | Multimodal neural models |
| 07 | `07_fit_state_space_process_model_generic.py` | State-space dynamic modeling |
| 08 | `08_sem_style_longitudinal_generic.py` | SEM-style causal modeling |
| 09 | `09_build_multitask_table.py` | Build multitask dataset |
| 10 | `10_train_multitask_shared_bottom.py` | Multi-task learning model |
| - | `covar_adjust.py` | Covariate adjustment utilities |

---

## 📊 Data Requirements

Input data must be prepared in **longitudinal long format**, where each row represents:

- one individual  
- one time point  

All domains (genetic, environmental, brain, behavioral) must be aligned by:
- subject ID  
- time index

## Data access and responsible use

This repository distributes analysis code and does not include participant-level research data. Users must obtain input data through the appropriate authorized providers and comply with applicable data-use agreements and institutional requirements.

For reproducibility, record the cohort, data release/version, accession or DOI where available, preprocessing steps, and exact code commit used.

Generated files may contain participant identifiers, individual measurements, outcomes, or predictions. Do not upload these files to a public repository without authorization. The software license does not grant permission to redistribute external datasets.

## Validation status and updates

This repository contains research code that may include undetected errors or limitations. The current source-code review did not include end-to-end execution or independent reproduction of the associated study results.

Known issues identified during review include import/path inconsistencies and a fallback that uses the test set for model selection when a separate validation set is unavailable. These issues require correction and validation.

Users should independently assess suitability for their analysis. Corrections following further checks will be documented in the commit history or release notes, including effects on outputs when known.

## 📄 Citation

If you use this framework or code, please cite:

Wei, M., & Peng, Q. (2026).  
*DynoSys: A Dynamic Systems Framework for Multimodal Integration of Genetic, Environmental, and Neurobiological Signals.*  
arXiv:2605.02952 [q-bio.OT].  
https://doi.org/10.48550/arXiv.2605.02952



