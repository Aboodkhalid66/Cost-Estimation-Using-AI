import joblib
import numpy as np
import pandas as pd


# ============================================================
# LOAD MODEL
# ============================================================

MODEL_FILE = "random_forest_all_drivers_model.pkl"
FEATURES_FILE = "all_drivers_model_features.pkl"
MAPPING_FILE = "all_drivers_rating_mapping.pkl"
DOMAINS_FILE = "all_drivers_domains.pkl"

model = joblib.load(MODEL_FILE)
model_features = joblib.load(FEATURES_FILE)
rating_mapping = joblib.load(MAPPING_FILE)
domains = joblib.load(DOMAINS_FILE)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def choose_option(title, options):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    for i, option in enumerate(options, start=1):
        print(f"{i}. {option}")

    while True:
        try:
            choice = int(input("\nChoose number: "))

            if 1 <= choice <= len(options):
                return options[choice - 1]

            print("Please choose a valid number.")

        except ValueError:
            print("Please enter a number.")


def rating_input(title):
    options = [
        ("vl", "Very Low"),
        ("l", "Low"),
        ("n", "Nominal / Medium"),
        ("h", "High"),
        ("vh", "Very High"),
        ("xh", "Extra High")
    ]

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    for i, (_, name) in enumerate(options, start=1):
        print(f"{i}. {name}")

    while True:
        try:
            choice = int(input("\nChoose number: "))

            if 1 <= choice <= len(options):
                return options[choice - 1][0]

            print("Please choose a valid number.")

        except ValueError:
            print("Please enter a number.")


# ============================================================
# START
# ============================================================

print("\n")
print("=" * 70)
print("       NASA93 SOFTWARE PROJECT COST ESTIMATION")
print("=" * 70)

print("\nRandom Forest Model")
print("All Effort Drivers")
print("")


# ============================================================
# 1. PROJECT DOMAIN
# ============================================================

domain = choose_option(
    "Select Project Domain",
    domains
)


# ============================================================
# 2. PROJECT SIZE
# ============================================================

print("\n" + "=" * 60)
print("Project Size")
print("=" * 60)

while True:
    try:
        kloc = float(input("Enter project size (KLOC): "))

        if kloc > 0:
            break

        print("KLOC must be greater than 0.")

    except ValueError:
        print("Please enter a valid number.")


# ============================================================
# 3. EFFORT DRIVERS
# ============================================================

rely = rating_input(
    "System Reliability (RELY)"
)

data = rating_input(
    "Database Size / Data Requirements (DATA)"
)

cplx = rating_input(
    "Product Complexity (CPLX)"
)

time = rating_input(
    "Execution Time Constraint (TIME)"
)

stor = rating_input(
    "Main Storage Constraint (STOR)"
)

virt = rating_input(
    "Virtual Machine Volatility (VIRT)"
)

turn = rating_input(
    "Computer Turnaround Time (TURN)"
)

acap = rating_input(
    "Analyst Capability (ACAP)"
)

aexp = rating_input(
    "Application Experience (AEXP)"
)

pcap = rating_input(
    "Programmer Capability (PCAP)"
)

vexp = rating_input(
    "Virtual Machine Experience (VEXP)"
)

lexp = rating_input(
    "Programming Language Experience (LEXP)"
)

modp = rating_input(
    "Modern Programming Practices (MODP)"
)

tool = rating_input(
    "Software Tools Usage (TOOL)"
)

sced = rating_input(
    "Required Development Schedule (SCED)"
)


# ============================================================
# 4. BUILD INPUT DATA
# ============================================================

input_data = {
    "equivphyskloc": kloc,

    "rely": rating_mapping[rely],
    "data": rating_mapping[data],
    "cplx": rating_mapping[cplx],
    "time": rating_mapping[time],
    "stor": rating_mapping[stor],
    "virt": rating_mapping[virt],
    "turn": rating_mapping[turn],
    "acap": rating_mapping[acap],
    "aexp": rating_mapping[aexp],
    "pcap": rating_mapping[pcap],
    "vexp": rating_mapping[vexp],
    "lexp": rating_mapping[lexp],
    "modp": rating_mapping[modp],
    "tool": rating_mapping[tool],
    "sced": rating_mapping[sced]
}


# ============================================================
# 5. DOMAIN ONE-HOT ENCODING
# ============================================================

for feature in model_features:

    if feature.startswith("cat2_"):
        input_data[feature] = 0.0


domain_feature = "cat2_" + domain

if domain_feature in input_data:
    input_data[domain_feature] = 1.0


# ============================================================
# 6. CREATE DATAFRAME
# ============================================================

X_new = pd.DataFrame(
    [input_data],
    columns=model_features
)


# Make sure all values are numeric
X_new = X_new.apply(
    pd.to_numeric,
    errors="coerce"
)


# ============================================================
# 7. PREDICT LOG EFFORT
# ============================================================

predicted_log_effort = model.predict(X_new)[0]


# ============================================================
# 8. CONVERT LOG EFFORT BACK TO REAL EFFORT
# ============================================================

estimated_effort = np.expm1(
    predicted_log_effort
)


# Prevent negative result
estimated_effort = max(
    0,
    estimated_effort
)


# ============================================================
# 9. COST PER PERSON-MONTH
# ============================================================

print("\n" + "=" * 60)
print("Cost Information")
print("=" * 60)

while True:
    try:
        cost_per_person_month = float(
            input("Enter cost per person-month ($): ")
        )

        if cost_per_person_month >= 0:
            break

        print("Cost cannot be negative.")

    except ValueError:
        print("Please enter a valid number.")


# ============================================================
# 10. CALCULATE PROJECT COST
# ============================================================

estimated_cost = (
    estimated_effort *
    cost_per_person_month
)


# ============================================================
# 11. DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("                 ESTIMATION RESULTS")
print("=" * 70)

print(f"\nProject Domain       : {domain}")
print(f"Project Size        : {kloc:.2f} KLOC")

print("\n" + "-" * 70)

print(
    f"Estimated Effort    : "
    f"{estimated_effort:.2f} Person-Month"
)

print(
    f"Cost / Person-Month : "
    f"${cost_per_person_month:,.2f}"
)

print(
    f"Estimated Cost      : "
    f"${estimated_cost:,.2f}"
)

print("-" * 70)

print("\n")
print("=" * 70)
print("             ESTIMATION COMPLETED")
print("=" * 70)