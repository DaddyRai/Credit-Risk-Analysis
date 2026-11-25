
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# STEP 1: Locate data
file_path = (
    Path(__file__).resolve().parent
    / "data"
    / "accepted_2007_to_2018Q4.csv.gz"
)

columns = [
    "issue_d", "loan_status",
    "fico_range_low", "annual_inc",
    "dti", "loan_amnt", "term"
]

# STEP 2: Load historical loan records
results = []

for chunk in pd.read_csv(
    file_path,
    compression="gzip",
    usecols=columns,
    chunksize=25000
):
    chunk["issue_year"] = pd.to_datetime(
        chunk["issue_d"],
        format="%b-%Y",
        errors="coerce"
    ).dt.year

    chunk = chunk[
        chunk["issue_year"].between(2010, 2012)
        & chunk["loan_status"].isin(
            ["Fully Paid", "Charged Off"]
        )
    ].copy()

    if not chunk.empty:
        results.append(chunk)

loans = pd.concat(results, ignore_index=True)

# STEP 3: Create the target variable
loans["is_default"] = (
    loans["loan_status"] == "Charged Off"
).astype(int)

# STEP 4: Select borrower characteristics
numeric_features = [
    "fico_range_low",
    "annual_inc",
    "dti",
    "loan_amnt"
]

categorical_features = ["term"]

for col in numeric_features:
    loans[col] = pd.to_numeric(
        loans[col], errors="coerce"
    )

loans["term"] = (
    loans["term"].astype("string")
    .str.strip()
    .fillna("Unknown")
)

features = numeric_features + categorical_features

# STEP 5: Train on earlier years
train = loans[loans["issue_year"] < 2012]
test = loans[loans["issue_year"] == 2012]

X_train = train[features]
y_train = train["is_default"]

X_test = test[features]
y_test = test["is_default"]

# STEP 6: Prepare the input data
preprocessor = ColumnTransformer([
    (
        "numeric",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]),
        numeric_features
    ),
    (
        "categorical",
        OneHotEncoder(handle_unknown="ignore"),
        categorical_features
    )
])

# STEP 7: Build the model
model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# STEP 8: Train using historical outcomes
model.fit(X_train, y_train)

# STEP 9: Predict the 2012 loan outcomes
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

# STEP 10: Evaluate our model
print("\nCREDIT RISK MODEL RESULTS")

print("Training loans:", len(train))
print("Testing loans:", len(test))

print(
    "Actual test charge-off rate:",
    round(y_test.mean() * 100, 2), "%"
)

print(
    "ROC-AUC:",
    round(roc_auc_score(y_test, probabilities), 3)
)

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    zero_division=0
))


# PHASE 10: TEST DIFFERENT PREDICTION THRESHOLDS

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

# Test several possible classification thresholds
thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.50]

print("\nTHRESHOLD COMPARISON")

for threshold in thresholds:

    # Step 1: Convert probabilities into predictions
    new_predictions = (
        probabilities >= threshold
    ).astype(int)

    # Step 2: Calculate evaluation metrics
    precision = precision_score(
        y_test, new_predictions, zero_division=0
    )

    recall = recall_score(
        y_test, new_predictions, zero_division=0
    )

    f1 = f1_score(
        y_test, new_predictions, zero_division=0
    )

    # Step 3: Examine correct and incorrect predictions
    tn, fp, fn, tp = confusion_matrix(
        y_test, new_predictions, labels=[0, 1]
    ).ravel()

    # Step 4: Display findings
    print(f"\nThreshold: {threshold:.0%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"F1 Score: {f1:.3f}")
    print(f"Defaults correctly flagged: {tp:,}")
    print(f"Defaults missed: {fn:,}")
    print(f"Good loans incorrectly flagged: {fp:,}")


# PHASE 11: VALIDATION-BASED THRESHOLD SELECTION

# Step 1: Separate training and validation years
train_2010 = loans[
    loans["issue_year"] == 2010
]

validation_2011 = loans[
    loans["issue_year"] == 2011
]

# Step 2: Define our inputs and actual outcomes
X_train_new = train_2010[features]
y_train_new = train_2010["is_default"]

X_val = validation_2011[features]
y_val = validation_2011["is_default"]

# Step 3: Retrain our existing model using 2010 only
model.fit(X_train_new, y_train_new)

# Step 4: Predict default probabilities for 2011
val_probabilities = model.predict_proba(X_val)[:, 1]

# Step 5: Compare thresholds
thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]

print("\n2011 VALIDATION RESULTS")

for threshold in thresholds:

    predictions_val = (
        val_probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_val, predictions_val, zero_division=0
    )

    recall = recall_score(
        y_val, predictions_val, zero_division=0
    )

    f1 = f1_score(
        y_val, predictions_val, zero_division=0
    )

    print(
        f"Threshold: {threshold:.0%} | "
        f"Precision: {precision:.2%} | "
        f"Recall: {recall:.2%} | "
        f"F1: {f1:.3f}"
    )

# PHASE 12: EVALUATE OUR SELECTED THRESHOLD

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Step 1: Lock in our selected threshold
best_threshold = 0.15

# Step 2: Combine 2010 and 2011 for training
final_train = loans[
    loans["issue_year"].isin([2010, 2011])
]

# Step 3: Keep 2012 for evaluation
final_test = loans[
    loans["issue_year"] == 2012
]

# Step 4: Define features and targets
X_final_train = final_train[features]
y_final_train = final_train["is_default"]

X_final_test = final_test[features]
y_final_test = final_test["is_default"]

# Step 5: Retrain with both earlier years
model.fit(X_final_train, y_final_train)

# Step 6: Generate predicted probabilities
final_probabilities = model.predict_proba(
    X_final_test
)[:, 1]

# Step 7: Apply our chosen threshold
final_predictions = (
    final_probabilities >= best_threshold
).astype(int)

# Step 8: Calculate performance metrics
precision = precision_score(
    y_final_test, final_predictions,
    zero_division=0
)

recall = recall_score(
    y_final_test, final_predictions,
    zero_division=0
)

f1 = f1_score(
    y_final_test, final_predictions,
    zero_division=0
)

auc = roc_auc_score(
    y_final_test, final_probabilities
)

# Step 9: Calculate confusion matrix
tn, fp, fn, tp = confusion_matrix(
    y_final_test,
    final_predictions,
    labels=[0, 1]
).ravel()

# Step 10: Display our results
print("\nFINAL MODEL EVALUATION")
print(f"Selected threshold: {best_threshold:.0%}")
print(f"ROC-AUC: {auc:.3f}")
print(f"Precision: {precision:.2%}")
print(f"Recall: {recall:.2%}")
print(f"F1 Score: {f1:.3f}")

print("\nCONFUSION MATRIX")
print(f"Defaults correctly flagged: {tp:,}")
print(f"Defaults missed: {fn:,}")
print(f"Good loans incorrectly flagged: {fp:,}")
print(f"Good loans correctly identified: {tn:,}")


# PHASE 13: MODEL PROBABILITY CALIBRATION

# Step 1: Store predicted probabilities and actual outcomes
calibration_data = pd.DataFrame({
    "predicted_pd": final_probabilities,
    "actual_default": y_final_test.to_numpy()
})

# Step 2: Divide borrowers into 10 risk groups
# Group 1 = lowest predicted risk
# Group 10 = highest predicted risk

calibration_data["risk_decile"] = pd.qcut(
    calibration_data["predicted_pd"],
    q=10,
    labels=False,
    duplicates="drop"
) + 1

# Step 3: Compare predicted and actual default rates
calibration_summary = calibration_data.groupby(
    "risk_decile"
).agg(
    loans=("actual_default", "size"),
    avg_predicted_pd=("predicted_pd", "mean"),
    actual_default_rate=("actual_default", "mean")
)

# Step 4: Convert proportions to percentages
calibration_summary["avg_predicted_pd"] *= 100
calibration_summary["actual_default_rate"] *= 100

# Step 5: Calculate prediction error
calibration_summary["difference_pp"] = (
    calibration_summary["actual_default_rate"]
    - calibration_summary["avg_predicted_pd"]
)

# Step 6: Display results
print("\nMODEL CALIBRATION BY RISK DECILE")

print(
    calibration_summary.round(2).to_string()
)

# Step 7: Compare average estimated risk to actual risk
print("\nOVERALL CALIBRATION")

print(
    "Average predicted default rate:",
    f"{calibration_data['predicted_pd'].mean():.2%}"
)

print(
    "Actual charge-off rate:",
    f"{calibration_data['actual_default'].mean():.2%}"
)


# PHASE 14: FEATURE IMPORTANCE

from sklearn.inspection import permutation_importance
from sklearn.metrics import brier_score_loss

# Step 1: Measure how accurate our probabilities are
brier = brier_score_loss(
    y_final_test,
    final_probabilities
)

# Compare against always predicting the
# historical charge-off rate from training data
baseline_probability = y_final_train.mean()

baseline_brier = brier_score_loss(
    y_final_test,
    [baseline_probability] * len(y_final_test)
)

print("\nPROBABILITY ACCURACY")
print(f"Model Brier Score: {brier:.4f}")
print(f"Baseline Brier Score: {baseline_brier:.4f}")

# Lower Brier scores are better

# Step 2: Measure feature importance
importance_results = permutation_importance(
    model,
    X_final_test,
    y_final_test,
    scoring="roc_auc",
    n_repeats=3,
    random_state=42
)

# Step 3: Create a readable results table
feature_importance = pd.DataFrame({
    "feature": X_final_test.columns,
    "importance": importance_results.importances_mean
})

# Step 4: Rank features by importance
feature_importance = feature_importance.sort_values(
    by="importance",
    ascending=False
)

# Step 5: Display results
print("\nFEATURE IMPORTANCE RESULTS")
print(feature_importance.round(4).to_string(index=False))


# PHASE 15: EXPORT MODEL RESULTS FOR POWER BI

# Step 1: Copy our 2012 test data
scored_loans = X_final_test.reset_index(drop=True).copy()

# Step 2: Add actual loan outcomes
scored_loans["actual_default"] = (
    y_final_test.reset_index(drop=True)
)

# Step 3: Add predicted default probabilities
scored_loans["predicted_pd"] = final_probabilities

# Step 4: Add our 15% risk classification
scored_loans["risk_flag"] = final_predictions

# Step 5: Divide borrowers into risk deciles
scored_loans["risk_decile"] = (
    pd.qcut(
        scored_loans["predicted_pd"],
        q=10,
        labels=False,
        duplicates="drop"
    ) + 1
)

# Step 6: Set the output file location
output_path = (
    Path(__file__).resolve().parent
    / "data"
    / "scored_loans_2012.csv"
)

# Step 7: Save as CSV
scored_loans.to_csv(output_path, index=False)

print("\nPOWER BI DATASET CREATED")
print("Rows exported:", len(scored_loans))
print("Saved to:", output_path)

# Display a preview
print("\nFIRST FIVE SCORED LOANS")
print(scored_loans.head().to_string())


# PHASE 16: EXPORT MODEL DIAGNOSTICS

output_dir = Path(__file__).resolve().parent / "data"

# 1. Calibration data
calibration_export = calibration_summary.reset_index().copy()

# Convert percentages back to decimal fractions for Power BI
calibration_export["avg_predicted_pd"] /= 100
calibration_export["actual_default_rate"] /= 100

calibration_export.to_csv(
    output_dir / "calibration_by_decile.csv",
    index=False
)

# 2. Feature importance
feature_importance.to_csv(
    output_dir / "feature_importance.csv",
    index=False
)

# 3. Brier score comparison
brier_comparison = pd.DataFrame({
    "model": ["Logistic Regression", "Baseline"],
    "brier_score": [brier, baseline_brier]
})

brier_comparison.to_csv(
    output_dir / "brier_comparison.csv",
    index=False
)

# 4. Classification outcomes
outcomes = pd.DataFrame({
    "outcome": [
        "Defaults Detected",
        "Defaults Missed",
        "Good Loans Flagged",
        "Good Loans Identified"
    ],
    "count": [tp, fn, fp, tn]
})

outcomes.to_csv(
    output_dir / "classification_outcomes.csv",
    index=False
)

print("\nDashboard diagnostic files exported!")
