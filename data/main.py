

import pandas as pd

# Location of our compressed dataset
file_path = "data/accepted_2007_to_2018Q4.csv.gz"

# Load the full dataset, selecting only the columns we need
columns_needed = [
    "grade",
    "term",
    "loan_status",
    "int_rate",
    "funded_amnt",
    "total_pymnt",
    "issue_d"
]

loans = pd.read_csv(
    file_path,
    compression="gzip",
    usecols=columns_needed,
    low_memory=False
)

print("Total records loaded:", len(loans))


# Display the first 5 loans
print("\nFirst 5 loans:")
print(loans.head())

# Display number of rows and columns
print("\nDataset size:")
print(loans.shape)

# Count loans by repayment status
print("\nLoan statuses:")
print(loans["loan_status"].value_counts())


# Select only loans with completed outcomes
finished_loans = loans[
    loans["loan_status"].isin(["Fully Paid", "Charged Off"])
]

# Count charged-off loans
defaults = (
    finished_loans["loan_status"] == "Charged Off"
).sum()

# Count all finished loans
total_loans = len(finished_loans)

# Calculate default rate
default_rate = (defaults / total_loans) * 100

print("\nCREDIT RISK ANALYSIS")
print("Completed loans:", total_loans)
print("Charged-off loans:", defaults)
print(f"Default rate: {default_rate:.2f}%")

# Create a seperate copy of completed loans for further analysis
risk_data = finished_loans.copy()

# Mark charged-off loans as True
risk_data["is_default"] = ( 
    risk_data["loan_status"] == "Charged Off"
)
 # Calculate the default rate by credit score
grade_analysis = risk_data.groupby("grade")["is_default"].agg(
    ["count", "mean"]
)

# Convert into a percentage
grade_analysis[default_rate] = (grade_analysis["mean"] * 100
).round(2)

print("\nDefault rate by credit grade:")
print(grade_analysis[["count", default_rate]])


# Compare different credit approval policies

policies = {
    "All Grades": ["A", "B", "C", "D", "E", "F", "G"],
    "Grades A-C": ["A", "B", "C"],
    "Grades A-B": ["A", "B"]
}

for policy_name, grades in policies.items():

    # Keep only loans belonging to the selected grades
    selected = risk_data[
        risk_data["grade"].isin(grades)
    ]

    # Calculate charge-off rate
    default_rate = selected["is_default"].mean() * 100

    # Calculate percentage of original loans retained
    retention = len(selected) / len(risk_data) * 100

    print(f"\nPolicy: {policy_name}")
    print(f"Loans retained: {len(selected):,}")
    print(f"Portfolio retained: {retention:.2f}%")
    print(f"Charge-off rate: {default_rate:.2f}%")


# PHASE 2: RISK VS RETURN ANALYSIS

# Step 1: Keep loans with completed outcomes
risk_data = loans[
    loans["loan_status"].isin(["Fully Paid", "Charged Off"])
].copy()

# Step 2: Create a default indicator
# 1 = Charged Off, 0 = Fully Paid
risk_data["is_default"] = (
    risk_data["loan_status"] == "Charged Off"
).astype(int)

# Step 3: Convert financial columns into numbers
for column in ["funded_amnt", "total_pymnt"]:
    risk_data[column] = pd.to_numeric(
        risk_data[column], errors="coerce"
    )

risk_data["int_rate"] = pd.to_numeric(
    risk_data["int_rate"].astype(str).str.replace(
        "%", "", regex=False
    ),
    errors="coerce"
)

# Remove incomplete financial records
risk_data = risk_data.dropna(
    subset=["grade", "int_rate", "funded_amnt", "total_pymnt"]
)
risk_data = risk_data[risk_data["funded_amnt"] > 0]

# Step 4: Group loans by credit grade
summary = risk_data.groupby("grade").agg(
    loan_count=("is_default", "size"),
    defaults=("is_default", "sum"),
    avg_interest=("int_rate", "mean"),
    funded=("funded_amnt", "sum"),
    collected=("total_pymnt", "sum")
)

# Step 5: Calculate financial performance
summary["default_pct"] = (
    summary["defaults"] / summary["loan_count"] * 100
)

summary["cash_surplus"] = (
    summary["collected"] - summary["funded"]
)

summary["cash_surplus_pct"] = (
    summary["cash_surplus"] / summary["funded"] * 100
)

# Step 6: Display results
print("\nRISK VS RETURN BY CREDIT GRADE")

print(
    summary[
        ["loan_count", "default_pct",
         "avg_interest", "cash_surplus_pct"]
    ].round(2).to_string()
)


# Compare three possible lending strategies

policies = {
    "All Grades": ["A", "B", "C", "D", "E", "F", "G"],
    "Grades A-C": ["A", "B", "C"],
    "Grades A-B": ["A", "B"]
}

for name, grades in policies.items():

    selected = risk_data[
        risk_data["grade"].isin(grades)
    ]

    funded = selected["funded_amnt"].sum()
    collected = selected["total_pymnt"].sum()

    surplus = collected - funded
    surplus_pct = surplus / funded * 100

    print(f"\nStrategy: {name}")
    print(f"Loans: {len(selected):,}")
    print(f"Cash surplus: ${surplus:,.2f}")
    print(f"Cash surplus per $100 funded: ${surplus_pct:.2f}")


# PHASE 3: HIGH-RISK PORTFOLIO ANALYSIS

# Step 1: Select high-risk loans
high_risk = risk_data[
    risk_data["grade"].isin(["D", "E", "F", "G"])
].copy()

# Step 2: Calculate cash surplus per loan
high_risk["cash_surplus"] = (
    high_risk["total_pymnt"]
    - high_risk["funded_amnt"]
)

# Step 3: Group results by credit grade
high_risk_summary = high_risk.groupby("grade").agg(
    loans=("grade", "size"),
    total_funded=("funded_amnt", "sum"),
    total_collected=("total_pymnt", "sum"),
    total_surplus=("cash_surplus", "sum")
)

# Step 4: Calculate surplus per $100 funded
high_risk_summary["surplus_per_100"] = (
    high_risk_summary["total_surplus"]
    / high_risk_summary["total_funded"]
    * 100
)

# Step 5: Display our findings
print("\nHIGH-RISK PORTFOLIO BREAKDOWN")
print(high_risk_summary.round(2).to_string())


# PHASE 4: INVESTIGATING GRADE E

# Step 1: Select only Grade E borrowers
grade_e = risk_data[
    risk_data["grade"] == "E"
].copy()

# Step 2: Clean the loan term column
grade_e["term"] = (
    grade_e["term"].astype(str).str.strip()
)

# Step 3: Group loans by term and repayment outcome
term_analysis = grade_e.groupby(
    ["term", "loan_status"]
).agg(
    loan_count=("grade", "size"),
    avg_interest=("int_rate", "mean"),
    total_funded=("funded_amnt", "sum"),
    total_collected=("total_pymnt", "sum")
)

# Step 4: Calculate cash surplus
term_analysis["cash_surplus"] = (
    term_analysis["total_collected"]
    - term_analysis["total_funded"]
)

# Step 5: Print the results
print("\nGRADE E: LOAN TERM ANALYSIS")
print(term_analysis.round(2).to_string())
