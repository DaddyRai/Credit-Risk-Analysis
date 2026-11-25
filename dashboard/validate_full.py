
import pandas as pd
from pathlib import Path

# STEP 1: Locate our dataset
project_folder = Path(__file__).resolve().parent

file_path = (
    project_folder
    / "data"
    / "accepted_2007_to_2018Q4.csv.gz"
)

# Verify that the file exists
if not file_path.exists():
    raise FileNotFoundError(f"Dataset not found: {file_path}")

# STEP 2: Select only necessary columns
columns = [
    "grade",
    "term",
    "loan_status",
    "funded_amnt",
    "total_pymnt",
    "int_rate",
    "issue_d"
]

results = []
total_rows = 0

print("Starting full dataset analysis...")

# STEP 3: Read the dataset in smaller chunks
try:
    reader = pd.read_csv(
        file_path,
        compression="gzip",
        usecols=columns,
        chunksize=25000,
        low_memory=False
    )

    for chunk_number, chunk in enumerate(reader, start=1):

        total_rows += len(chunk)

        # Select completed Grade E loans
        grade_e = chunk[
            (chunk["grade"] == "E")
            & (chunk["loan_status"].isin(
                ["Fully Paid", "Charged Off"]
            ))
        ].copy()

        if grade_e.empty:
            continue

        # Clean loan terms
        grade_e["term"] = (
            grade_e["term"].astype(str).str.strip()
        )

        # Convert origination dates to years
        grade_e["issue_year"] = pd.to_datetime(
            grade_e["issue_d"],
            format="%b-%Y",
            errors="coerce"
        ).dt.year

        # Convert financial fields to numbers
        for col in ["funded_amnt", "total_pymnt", "int_rate"]:
            grade_e[col] = pd.to_numeric(
                grade_e[col],
                errors="coerce"
            )

        # Remove incomplete records
        grade_e = grade_e.dropna(
            subset=[
                "term",
                "issue_year",
                "funded_amnt",
                "total_pymnt",
                "int_rate"
            ]
        )

        grade_e = grade_e[
            (grade_e["funded_amnt"] > 0)
            & (grade_e["term"].isin(
                ["36 months", "60 months"]
            ))
        ].copy()

        if grade_e.empty:
            continue

        # Label charged-off loans
        grade_e["is_default"] = (
            grade_e["loan_status"] == "Charged Off"
        ).astype(int)

        # Summarize the chunk
        group = grade_e.groupby(
            ["issue_year", "term"]
        ).agg(
            loans=("is_default", "size"),
            defaults=("is_default", "sum"),
            funded=("funded_amnt", "sum"),
            collected=("total_pymnt", "sum"),
            interest_sum=("int_rate", "sum")
        ).reset_index()

        results.append(group)

        # Show progress every 10 chunks
        if chunk_number % 10 == 0:
            print(f"Processed {total_rows:,} rows")

except Exception as error:
    print("\nERROR WHILE READING DATA")
    print("Error type:", type(error).__name__)
    print("Error details:", error)
    raise

# STEP 4: Combine results
if not results:
    raise ValueError("No valid Grade E loans found.")

combined = pd.concat(results, ignore_index=True)

print(f"\nTotal records processed: {total_rows:,}")


# STEP 5: Function for financial analysis
def show_report(data, title):

    summary = data.groupby("term").agg(
        loans=("loans", "sum"),
        defaults=("defaults", "sum"),
        funded=("funded", "sum"),
        collected=("collected", "sum"),
        interest_sum=("interest_sum", "sum")
    )

    summary["default_rate"] = (
        summary["defaults"] / summary["loans"] * 100
    )

    summary["avg_interest"] = (
        summary["interest_sum"] / summary["loans"]
    )

    summary["surplus_per_100"] = (
        (summary["collected"] - summary["funded"])
        / summary["funded"] * 100
    )

    print(f"\n{title}")

    print(
        summary[
            ["loans", "default_rate",
             "avg_interest", "surplus_per_100"]
        ].round(2).to_string()
    )


# STEP 6: Full dataset report
show_report(
    combined,
    "FULL DATASET: GRADE E"
)

# STEP 7: Historical cohort comparison
older = combined[
    combined["issue_year"] <= 2013
]

show_report(
    older,
    "2013 AND EARLIER: GRADE E"
)

# STEP 8: Year-by-year comparison
cohorts = combined[
    combined["issue_year"].between(2010, 2012)
]

year_analysis = cohorts.groupby(
    ["issue_year", "term"]
)[
    ["loans", "defaults", "funded", "collected"]
].sum().reset_index()

year_analysis["default_rate"] = (
    year_analysis["defaults"]
    / year_analysis["loans"] * 100
)

year_analysis["surplus_per_100"] = (
    (year_analysis["collected"] - year_analysis["funded"])
    / year_analysis["funded"] * 100
)

print("\nGRADE E: YEAR-BY-YEAR COHORT ANALYSIS")

print(
    year_analysis[
        ["issue_year", "term", "loans",
         "default_rate", "surplus_per_100"]
    ].round(2).to_string(index=False)
)

# STEP 9: Compare 2012 and earlier
show_report(
    combined[combined["issue_year"] <= 2012],
    "GRADE E: 2012 AND EARLIER"
)

print("\nAnalysis completed successfully!")
