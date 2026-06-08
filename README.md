# Student Dropout Determinants and Early Warning Modeling

This repository contains the full research workflow for analyzing university student dropout determinants and building early warning models.

## Project Structure

- `data/`: dataset used in the analysis.
- `models/`: reusable Python scripts for preprocessing, model training, tuning, SHAP, clustering, and graph analysis.
- `results/`: generated analysis outputs, tables, figures, reports, trained models, and experiment artifacts.
- `Experiments&Methodology/`: paper-ready selected tables, figures, and Word sections for methodology, experiments, findings, conclusion, and research questions.
- `1_EDA.ipynb` to `9_Graph_Analysis.ipynb`: step-by-step notebooks for the research pipeline.
- `Early Warning Signals_03.06.docx`: original paper document.

## Analysis Pipeline

1. Exploratory data analysis
2. Modeling preprocessing and feature set definition
3. Staged early warning modeling
4. Hyperparameter tuning
5. Full-feature final modeling
6. Ablation study
7. Feature importance and SHAP interpretation
8. Risk clustering
9. Student and course graph analysis

## Main Research Aim

The project studies which academic, demographic, financial, parental, macroeconomic, course, and application-related factors are associated with student dropout. The goal is to support universities with interpretable early warning evidence and practical intervention groups.

## Paper-Ready Outputs

The folder `Experiments&Methodology/` contains the selected tables and figures recommended for the paper, plus drafted Word sections:

- `Methodology_Section.docx`
- `Experiments_and_Results_Section.docx`
- `Research_Findings_and_Conclusion.docx`
- `Research_Questions_and_Addressing_Strategy.docx`
