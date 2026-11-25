# Student Dropout Determinants and Early Warning Modeling

## Project Overview

This repository contains the research workflow used to study the factors associated with university student dropout and to develop early warning models based on student-level data.

The project combines exploratory data analysis, predictive modeling, feature interpretation, clustering, and graph-based analysis. The main purpose is to understand which academic, demographic, financial, family-related, macroeconomic, course, and application-related factors are associated with dropout and whether these factors can provide useful early-warning signals.

The repository also contains additional analyses and experiments that were completed during the revision of the study. These revised analyses should be used for the current results.

## Revised Analyses and Additional Experiments

The revised analysis is mainly contained in the following folders and files:

- `Supplementary Analysis S1 - 09_10/` contains the updated predictive refits using the recorded settings, comparisons between different stages of the early-warning process, financial-timing sensitivity analyses, SHAP summaries, clustering checks, source inputs, and reproducibility instructions.
- `llm_study/` contains the experiments related to generating explanations from model predictions and selected SHAP evidence. This includes the explanation-generation scripts, frozen prompts and evaluation protocol, decoded student evidence, generated and template-based explanations, and the corresponding evaluation tables and plots. The README in this folder provides the starting point for reproducing these experiments.
- `models/evaluation.py` contains the corrected implementation of balanced accuracy, which is now calculated as macro-averaged recall.

For the updated results, the revised analysis packages should be used. The notebooks and original outputs included in the repository are retained as a historical snapshot and should not be mixed with the revised results.

The program-similarity graph is not included in the revised evidence because its construction relied on outcome information. It was therefore excluded from the updated analysis.

## Explanation Experiments

The LLM-based explanation study is designed as a post-hoc explanation step rather than a prediction method.

The model receives an already generated prediction together with selected SHAP evidence and the direction of the corresponding feature effects. It then converts this information into a natural-language explanation.

Therefore, the LLM does not identify new causes, discover previously unknown relationships, or improve the predictive performance of the underlying models. Its role is limited to verbalizing the model evidence that is provided to it.

The explanation evaluation was conducted automatically after the explanations were generated. No human evaluation of clarity, usefulness, or quality was collected. For this reason, the results should be interpreted as an automated assessment of the generated explanations rather than as evidence of their usefulness to students, advisors, or university staff.

Similarly, the project should not be interpreted as a prospectively validated early-warning system. The results provide research evidence about predictive performance and associated factors, but further validation would be required before using such a system in a real university setting.

## Historical Project Snapshot

The original repository structure and outputs are retained to document the development of the study and to make the earlier analysis traceable.

The main components are:

- `data/` — dataset used for the analysis.
- `models/` — reusable Python scripts for preprocessing, model training, hyperparameter tuning, SHAP analysis, clustering, and graph analysis.
- `results/` — generated results, tables, figures, reports, trained models, and other experiment artifacts.
- `Experiments&Methodology/` — selected paper-ready tables, figures, and drafted sections covering the methodology, experiments, findings, conclusion, and research questions.
- `1_EDA.ipynb` to `9_Graph_Analysis.ipynb` — notebooks documenting the analysis steps.
- `Early Warning Signals_03.06.docx` — the original paper document.

## Analysis Pipeline

The research workflow follows these main stages:

1. Exploratory data analysis
2. Data preprocessing and definition of the modeling feature sets
3. Staged early-warning modeling
4. Hyperparameter tuning
5. Final modeling using the full feature set
6. Ablation experiments
7. Feature importance and SHAP analysis
8. Risk clustering
9. Student and course graph analysis

These stages were used to examine both the predictive performance of the models and the factors contributing to the predictions.

## Main Research Aim

The main aim of the project is to investigate which factors are associated with university student dropout and to examine whether these factors can be used to identify students who may be at higher risk at an earlier stage of their studies.

The analysis considers a broad range of information, including academic performance, demographic characteristics, financial circumstances, parental background, macroeconomic conditions, course-related information, and application-related factors.

An additional goal is to provide interpretable evidence that can help identify different risk profiles and potentially support future research into targeted university interventions.

## Paper-Ready Materials

The `Experiments&Methodology/` folder contains the selected materials prepared for the paper, including tables, figures, and drafted sections.

The main documents are:

- `Methodology_Section.docx`
- `Experiments_and_Results_Section.docx`
- `Research_Findings_and_Conclusion.docx`
- `Research_Questions_and_Addressing_Strategy.docx`

These files contain the current selected presentation of the methodology, experimental results, research findings, conclusions, and the strategy used to address the research questions.

## Reproducibility and Scope

The repository is intended to provide a traceable record of the research process, including the revised analyses and the earlier experimental work.

The revised analysis should be treated as the basis for the current reported results. Historical notebooks and outputs are kept for reference and documentation but should not be used interchangeably with the revised results.

The repository does not claim that the developed models constitute a validated operational early-warning system. Rather, it documents an empirical study of dropout-related factors, predictive modeling, and post-hoc model interpretation.
