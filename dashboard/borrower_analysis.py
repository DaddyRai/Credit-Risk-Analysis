
import pandas as pd
from pathlib import Path

# Locate our dataset
file_path = (
    Path(__file__).resolve().parent
    / "data"
    / "accepted_2007_to_2018Q4.csv.gz"
)

# Columns we want to investigate
columns = [
    "grade",
    "term",
    "loan_status",
    "issue_d",
    "fico_range_low",
    "annual_inc",
    "dti"
]

results = []

# Read 25,000 rows at a time
for chunk in pd.read_csv(
    file_path,
    compression="gzip",
    usecols=columns,
    chunksize=25000
):

    # Focus on Grade E loans
    chunk = chunk[
        (chunk["grade"] == "E")
        & chunk["loan_status"].isin(
            ["Fully Paid", "Charged Off"]
        )
    ].copy()

    # Convert issue date into year
    chunk["issue_year"] = pd.to_datetime(
        chunk["issue_d"],
        format="%b-%Y",
        errors="coerce"
    ).dt.year

    # Select mature historical cohorts
    chunk = chunk[
        chunk["issue_year"].between(2010, 2012)
    ].copy()

    if not chunk.empty:
        results.append(chunk)

# Combine the filtered loan records
borrowers = pd.concat(results, ignore_index=True)

# Create our binary default variable
borrowers["is_default"] = (
    borrowers["loan_status"] == "Charged Off"
).astype(int)

# Convert borrower characteristics to numbers
for col in ["fico_range_low", "annual_inc", "dti"]:
    borrowers[col] = pd.to_numeric(
        borrowers[col],
        errors="coerce"
    )

# Compare borrower characteristics by loan term
borrowers["term"] = borrowers["term"].str.strip()

summary = borrowers.groupby("term").agg(
    loan_count=("is_default", "size"),
    default_rate=("is_default", "mean"),
    avg_fico=("fico_range_low", "mean"),
    median_income=("annual_inc", "median"),
    median_dti=("dti", "median")
)

summary["default_rate"] *= 100

print("\nBORROWER RISK CHARACTERISTICS")
print(summary.round(2).to_string())


# PHASE 7: CREDIT SCORE SEGMENTATION

# Step 1: Remove missing FICO values
fico_data = borrowers.dropna(
    subset=["fico_range_low"]
).copy()

# Step 2: Create credit score groups
fico_data["fico_group"] = pd.cut(
    fico_data["fico_range_low"],
    bins=[0, 660, 680, 700, 720, float("inf")],
    labels=[
        "Below 660",
        "660-679",
        "680-699",
        "700-719",
        "720+"
    ],
    right=False
)

# Step 3: Compare loan terms within FICO groups
fico_analysis = fico_data.groupby(
    ["fico_group", "term"],
    observed=True
).agg(
    loan_count=("is_default", "size"),
    default_rate=("is_default", "mean")
).reset_index()

# Step 4: Convert default rates to percentages
fico_analysis["default_rate"] = (
    fico_analysis["default_rate"] * 100
).round(2)

# Step 5: Display results
print("\nCREDIT RISK BY FICO GROUP AND LOAN TERM")

print(fico_analysis.to_string(index=False))


# PHASE 8: CONTROL FOR FICO AND ORIGINATION YEAR

# Step 1: Select borrowers with FICO scores 660-699
focus = borrowers[
    (borrowers["fico_range_low"] >= 660) &
    (borrowers["fico_range_low"] < 700)
].copy()

# Step 2: Calculate default rate by year and term
year_comparison = focus.groupby(
    ["issue_year", "term"]
).agg(
    loans=("is_default", "size"),
    defaults=("is_default", "sum"),
    default_rate=("is_default", "mean")
).reset_index()

# Step 3: Convert default rate to percentage
year_comparison["default_rate"] *= 100

# Step 4: Display the results
print("\nFICO 660-699: YEAR-BY-YEAR ANALYSIS")

print(
    year_comparison.round(2).to_string(index=False)
)

# Step 5: Compare the default rates side by side
comparison = year_comparison.pivot(
    index="issue_year",
    columns="term",
    values="default_rate"
)

# Step 6: Calculate the difference
comparison["risk_gap_pp"] = (
    comparison["60 months"] -
    comparison["36 months"]
)

print("\nDEFAULT RATE DIFFERENCE BY YEAR")
print(comparison.round(2).to_string())
