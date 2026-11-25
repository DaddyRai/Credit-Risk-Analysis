
# Credit Risk & Loan Default Prediction

## Project Overview

This project investigates consumer loan default risk
using historical LendingClub lending data.

The goal is to identify borrower characteristics
associated with charge offs, build a predictive
machine learning model, and visualize findings
through an interactive Power BI dashboard.

## Tools & Technologies

- Python: Pandas, NumPy, Scikit-learn
- Machine Learning: Logistic Regression
- Power BI: DAX, Power Query, Data Visualization
- Analysis: Credit Risk, Cohort Analysis, Calibration

## Dataset

Source: LendingClub Loan Data (2007–2018)
https://www.kaggle.com/datasets/wordsforthewise/lending-club

Approximately 2.26 million historical records
were processed during exploratory analysis.

The predictive model was evaluated on 53,367
loans originated in 2012 with completed outcomes.

## Methodology

1. Explored loan repayment outcomes and credit grades.
2. Compared charge off rates and cash collections.
3. Analyzed borrower characteristics and loan terms.
4. Built a logistic regression classification model.
5. Selected a 15% classification threshold using
   2011 validation data.
6. Evaluated model performance on 2012 loans.
7. Examined calibration and permutation importance.
8. Created two interactive Power BI report pages.

## Model Results

| Metric | Result |
|---|---:|
| ROC-AUC | 0.663 |
| Precision | 24.31% |
| Recall | 59.00% |
| F1 Score | 0.344 |
| Brier Score | 0.1295 |
| Baseline Brier Score | 0.1361 |
| Actual Charge-Off Rate | 16.20% |
| Average Predicted PD | 14.76% |

## Key Findings

- FICO score and loan term were the two most
  influential predictors in the initial model.
- The highest predicted-risk decile had an observed
  charge-off rate of 34.12%, compared with 5.83%
  in the lowest-risk decile.
- The model underestimated the overall observed
  charge-off rate by 1.44 percentage points.
- The model achieved a lower Brier score than
  the constant-probability baseline.

## Power BI Dashboard

### Page 1: Portfolio & Model Overview

[Portfolio Dashboard] 
<img width="781" height="459" alt="image" src="https://github.com/user-attachments/assets/4342fa4f-efee-4318-be3b-8e97e9502b40" />

### Page 2: Model Diagnostics 
<img width="782" height="443" alt="image" src="https://github.com/user-attachments/assets/7c78f697-930b-4c81-b142-684622d7ba68" />

[Model Diagnostics]

## How to Run the Project

1. Download the original dataset from Kaggle.
2. Place accepted_2007_to_2018Q4.csv.gz
   inside the data folder.
3. Install dependencies:

   py -m pip install -r requirements.txt

4. Run the analysis scripts:

   py validate_full.py
   py borrower_analysis.py
   py default_model.py

## Limitations

This is an exploratory historical model, not a
production-ready credit scoring system.

The analysis focuses on loans with completed outcomes,
which can introduce selection bias.

Model development involved exploratory examination
of the 2012 evaluation cohort, so these results
should not be described as a fully untouched test.

Financial analysis does not establish causal effects
or demonstrate that a lending strategy is profitable.

Further work should include out-of-time validation,
probability calibration, and credit loss modeling.
