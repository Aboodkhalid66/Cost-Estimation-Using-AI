import joblib
import numpy as np
import pandas as pd

print("=" * 70)
print("SOFTWARE PROJECT EFFORT & COST ESTIMATION")
print("CORRECTED MODEL")
print("=" * 70)

# ============================================================
# LOAD CORRECTED MODEL FILES
# ============================================================

model = joblib.load("corrected_effort_model.pkl")
model_features = joblib.load("corrected_model_features.pkl")
driver_weights = joblib.load("corrected_driver_weights.pkl")
domains = joblib.load("corrected_domains.pkl")

print("\nModel files loaded successfully.")

# ============================================================
# PROJECT TYPE
# ============================================================

print("\nProject Type:")

for i, domain in enumerate(domains, 1):
    print(f"{i}. {domain}")

while True:
    try:
        domain_choice = int(input("\nEnter project type number: "))

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
        kloc = float(input("\nProject Size (KLOC): "))

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
            choice = int(input("Enter rating (1-6): "))

            if choice in rating_options:
                driver_values[driver] = rating_options[choice]
                break

            print("Please enter a number from 1 to 6.")

        except ValueError:
            print("Please enter a valid number.")

# ============================================================
# BUILD INPUT DATA
# ============================================================

input_data = {}

# Project size
input_data["equivphyskloc"] = kloc

# Log project size
input_data["log_equivphyskloc"] = np.log1p(kloc)

# Apply individual driver weights
for driver in driver_names:
    input_data[driver] = driver_weights[driver][driver_values[driver]]

# Domain One-Hot Encoding
for domain in domains:
    column_name = f"cat2_{domain}"

    if domain == selected_domain:
        input_data[column_name] = 1
    else:
        input_data[column_name] = 0

# ============================================================
# PREPARE DATA
# ============================================================

X = pd.DataFrame([input_data])

# Add missing features if necessary
for feature in model_features:
    if feature not in X.columns:
        X[feature] = 0

# Exact feature order used during training
X = X[model_features]

# ============================================================
# PREDICT EFFORT
# ============================================================

log_effort_prediction = model.predict(X)[0]

# Convert log prediction back to Person-Month
estimated_effort = np.expm1(log_effort_prediction)

# Prevent negative result
estimated_effort = max(0, estimated_effort)

# ============================================================
# COST INPUT
# ============================================================

while True:
    try:
        cost_per_pm = float(
            input("\nCost per Person-Month ($): ")
        )

        if cost_per_pm >= 0:
            break

        print("Cost cannot be negative.")

    except ValueError:
        print("Please enter a valid number.")

# ============================================================
# CALCULATE COST
# ============================================================

estimated_cost = estimated_effort * cost_per_pm

# ============================================================
# RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("ESTIMATION RESULTS")
print("=" * 70)

print(f"\nProject Type       : {selected_domain}")
print(f"Project Size       : {kloc:.2f} KLOC")
print(f"Estimated Effort   : {estimated_effort:.2f} Person-Month")
print(f"Cost per PM        : ${cost_per_pm:.2f}")
print(f"Estimated Cost     : ${estimated_cost:,.2f}")

print("\n" + "=" * 70)
print("ESTIMATION COMPLETED")
print("=" * 70)