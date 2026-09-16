import joblib
import numpy as np
import pandas as pd

print("=" * 70)
print("SOFTWARE PROJECT EFFORT & COST ESTIMATION")
print("MONOTONIC MODEL")
print("=" * 70)

# ============================================================
# LOAD MODEL FILES
# ============================================================

model = joblib.load("monotonic_effort_model.pkl")
model_features = joblib.load("monotonic_model_features.pkl")
driver_weights = joblib.load("monotonic_driver_weights.pkl")
domain_columns = joblib.load("monotonic_domains.pkl")

print("\nModel files loaded successfully.")


# ============================================================
# GET DOMAIN NAMES
# ============================================================

domains = []

for column in domain_columns:
    if column.startswith("cat2_"):
        domains.append(column.replace("cat2_", ""))


# ============================================================
# PROJECT TYPE
# ============================================================

print("\nProject Type:")

for i, domain in enumerate(domains, 1):
    print(f"{i}. {domain}")

while True:
    try:
        domain_choice = int(
            input("\nEnter project type number: ")
        )

        if 1 <= domain_choice <= len(domains):
            selected_domain = domains[domain_choice - 1]
            break

        print("Invalid choice. Try again.")

    except ValueError:
        print("Please enter a number.")


# ============================================================
# PROJECT SIZE
# ============================================================

while True:
    try:
        kloc = float(
            input("\nProject Size (KLOC): ")
        )

        if kloc > 0:
            break

        print("KLOC must be greater than 0.")

    except ValueError:
        print("Please enter a valid number.")


# ============================================================
# EFFORT DRIVERS
# ============================================================

driver_names = [
    "rely",
    "data",
    "cplx",
    "time",
    "stor",
    "virt",
    "turn",
    "acap",
    "aexp",
    "pcap",
    "vexp",
    "lexp",
    "modp",
    "tool",
    "sced"
]


rating_options = {
    1: "vl",
    2: "l",
    3: "n",
    4: "h",
    5: "vh",
    6: "xh"
}


rating_names = {
    1: "Very Low",
    2: "Low",
    3: "Nominal",
    4: "High",
    5: "Very High",
    6: "Extra High"
}


rating_values = {
    "vl": 0.5,
    "l": 0.75,
    "n": 1.0,
    "h": 1.15,
    "vh": 1.30,
    "xh": 1.45
}


driver_values = {}


print("\n" + "=" * 70)
print("EFFORT DRIVERS")
print("=" * 70)


for driver in driver_names:

    print(f"\n{driver.upper()}")

    for number, name in rating_names.items():
        print(f"{number}. {name}")

    while True:

        try:

            choice = int(
                input("Enter rating (1-6): ")
            )

            if choice in rating_options:

                driver_values[driver] = (
                    rating_options[choice]
                )

                break

            print(
                "Please enter a number from 1 to 6."
            )

        except ValueError:

            print(
                "Please enter a valid number."
            )


# ============================================================
# BUILD INPUT DATA
# ============================================================

input_data = {}

# Project size
input_data["equivphyskloc"] = kloc

# Log project size
input_data["log_equivphyskloc"] = np.log1p(kloc)


# ============================================================
# ADD DRIVER FEATURES
# ============================================================

for driver in driver_names:

    rating = driver_values[driver]

    # Original rating feature
    input_data[driver] = rating_values[rating]

    # COCOMO weight feature
    weight_feature = f"{driver}_weight"

    input_data[weight_feature] = (
        driver_weights[driver][rating]
    )


# ============================================================
# DOMAIN ONE-HOT ENCODING
# ============================================================

for column in domain_columns:

    if column == f"cat2_{selected_domain}":
        input_data[column] = 1.0
    else:
        input_data[column] = 0.0


# ============================================================
# CREATE DATAFRAME
# ============================================================

X = pd.DataFrame([input_data])


# Add missing features if necessary
for feature in model_features:

    if feature not in X.columns:
        X[feature] = 0.0


# Exact feature order
X = X[model_features]


# Convert to numeric
X = X.apply(
    pd.to_numeric,
    errors="coerce"
)


X = X.replace(
    [np.inf, -np.inf],
    np.nan
)


X = X.fillna(0.0)


# ============================================================
# PREDICT EFFORT
# ============================================================

log_effort_prediction = model.predict(X)[0]


# Convert log prediction back to Person-Month
estimated_effort = np.expm1(
    log_effort_prediction
)


# Prevent negative result
estimated_effort = max(
    0.0,
    estimated_effort
)


# ============================================================
# COST
# ============================================================

while True:

    try:

        cost_per_pm = float(
            input(
                "\nCost per Person-Month ($): "
            )
        )

        if cost_per_pm >= 0:
            break

        print(
            "Cost cannot be negative."
        )

    except ValueError:

        print(
            "Please enter a valid number."
        )


# ============================================================
# CALCULATE COST
# ============================================================

estimated_cost = (
    estimated_effort * cost_per_pm
)


# ============================================================
# RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("ESTIMATION RESULTS")
print("=" * 70)

print(
    f"\nProject Type       : {selected_domain}"
)

print(
    f"Project Size       : {kloc:.2f} KLOC"
)

print(
    f"Estimated Effort   : "
    f"{estimated_effort:.2f} Person-Month"
)

print(
    f"Cost per PM        : "
    f"${cost_per_pm:.2f}"
)

print(
    f"Estimated Cost     : "
    f"${estimated_cost:,.2f}"
)

print("\n" + "=" * 70)
print("ESTIMATION COMPLETED")
print("=" * 70)