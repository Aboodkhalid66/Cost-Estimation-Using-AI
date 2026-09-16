import pandas as pd
import numpy as np
import joblib

from scipy.io import arff
from sklearn.model_selection import KFold, cross_validate
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# NASA93 - IMPROVED EFFORT ESTIMATION MODEL
# ============================================================

print("=" * 70)
print("NASA93 - IMPROVED EFFORT ESTIMATION MODEL")
print("=" * 70)


# ============================================================
# 1. LOAD DATA
# ============================================================

data, meta = arff.loadarff("nasa93.arff")

df = pd.DataFrame(data)

# Decode byte values
for col in df.select_dtypes([object]).columns:
    df[col] = df[col].apply(
        lambda x: x.decode("utf-8")
        if isinstance(x, bytes)
        else x
    )

print("\nProjects:", len(df))


# ============================================================
# 2. TARGET
# ============================================================

TARGET = "act_effort"

y = pd.to_numeric(
    df[TARGET],
    errors="coerce"
)

valid = y.notna()

df = df.loc[valid].copy()
y = y.loc[valid].reset_index(drop=True)


# ============================================================
# 3. FEATURES
# ============================================================

rating_features = [
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


rating_mapping = {
    "vl": 0.50,
    "l": 0.75,
    "n": 1.00,
    "h": 1.15,
    "vh": 1.30,
    "xh": 1.45
}


# ============================================================
# 4. BUILD NUMERIC FEATURES
# ============================================================

X = pd.DataFrame(index=df.index)

# Project size
X["equivphyskloc"] = pd.to_numeric(
    df["equivphyskloc"],
    errors="coerce"
)

# Log of project size
# Helps the model understand very large size differences
X["log_equivphyskloc"] = np.log1p(
    X["equivphyskloc"]
)


# ============================================================
# 5. EFFORT DRIVERS
# ============================================================

for col in rating_features:

    X[col] = (
        df[col]
        .astype(str)
        .str.lower()
        .str.strip()
        .map(rating_mapping)
    )

    X[col] = X[col].fillna(
        X[col].median()
    )


# Fill KLOC missing values
X["equivphyskloc"] = X["equivphyskloc"].fillna(
    X["equivphyskloc"].median()
)

X["log_equivphyskloc"] = X["log_equivphyskloc"].fillna(
    X["log_equivphyskloc"].median()
)


# ============================================================
# 6. DOMAIN ENCODING
# ============================================================

domain = (
    df["cat2"]
    .astype(str)
    .str.lower()
    .str.strip()
)

domain_encoded = pd.get_dummies(
    domain,
    prefix="cat2",
    drop_first=True
)

domain_encoded = domain_encoded.astype(float)

X = pd.concat(
    [
        X.reset_index(drop=True),
        domain_encoded.reset_index(drop=True)
    ],
    axis=1
)


# ============================================================
# 7. FEATURE INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("MODEL FEATURES")
print("=" * 70)

print("\nNumber of features:", X.shape[1])

for i, feature in enumerate(X.columns, 1):
    print(f"{i:2d}. {feature}")


# ============================================================
# 8. IMPROVED MODEL
# ============================================================

# HistGradientBoosting supports monotonic constraints.
#
# We force:
#
# More KLOC -> cannot produce lower predicted effort
#
# This addresses the problem found during manual testing.

monotonic_constraints = []

for feature in X.columns:

    if feature in [
        "equivphyskloc",
        "log_equivphyskloc"
    ]:
        monotonic_constraints.append(1)

    else:
        monotonic_constraints.append(0)


model = HistGradientBoostingRegressor(
    max_iter=300,
    learning_rate=0.05,
    max_leaf_nodes=15,
    min_samples_leaf=5,
    l2_regularization=1.0,
    random_state=42,
    monotonic_cst=monotonic_constraints
)


# ============================================================
# 9. LOG TARGET
# ============================================================

y_log = np.log1p(y)


# ============================================================
# 10. METRICS
# ============================================================

def rmse(y_true, y_pred):

    return np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )


scoring = {
    "MAE": make_scorer(
        mean_absolute_error
    ),

    "RMSE": make_scorer(
        rmse
    ),

    "R2": make_scorer(
        r2_score
    )
}


# ============================================================
# 11. 5-FOLD CROSS VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("5-FOLD CROSS VALIDATION")
print("=" * 70)

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


cv_results = cross_validate(
    model,
    X,
    y_log,
    cv=kf,
    scoring=scoring,
    return_train_score=False
)


print("\nCross Validation Results:")

for i in range(5):

    print(
        f"Fold {i + 1}: "
        f"MAE={cv_results['test_MAE'][i]:.4f}, "
        f"RMSE={cv_results['test_RMSE'][i]:.4f}, "
        f"R2={cv_results['test_R2'][i]:.4f}"
    )


mean_mae = cv_results["test_MAE"].mean()
mean_rmse = cv_results["test_RMSE"].mean()
mean_r2 = cv_results["test_R2"].mean()


print("\n" + "=" * 70)
print("AVERAGE RESULTS")
print("=" * 70)

print(f"\nMean MAE : {mean_mae:.4f}")
print(f"Mean RMSE: {mean_rmse:.4f}")
print(f"Mean R2  : {mean_r2:.4f}")


# ============================================================
# 12. TRAIN FINAL MODEL
# ============================================================

print("\n" + "=" * 70)
print("TRAINING FINAL MODEL")
print("=" * 70)

model.fit(
    X,
    y_log
)

print("\nFinal model trained successfully.")


# ============================================================
# 13. SAVE MODEL
# ============================================================

joblib.dump(
    model,
    "improved_effort_model.pkl"
)

joblib.dump(
    list(X.columns),
    "improved_model_features.pkl"
)

joblib.dump(
    rating_mapping,
    "improved_rating_mapping.pkl"
)

joblib.dump(
    sorted(domain.unique()),
    "improved_domains.pkl"
)


# ============================================================
# 14. SAVE RESULTS
# ============================================================

results = pd.DataFrame({
    "Metric": [
        "Mean MAE",
        "Mean RMSE",
        "Mean R2"
    ],

    "Value": [
        mean_mae,
        mean_rmse,
        mean_r2
    ]
})

results.to_csv(
    "improved_training_results.csv",
    index=False
)


# ============================================================
# 15. COMPLETED
# ============================================================

print("\n" + "=" * 70)
print("IMPROVED MODEL TRAINING COMPLETED")
print("=" * 70)

print("\nFiles created:")

print("1. improved_effort_model.pkl")
print("2. improved_model_features.pkl")
print("3. improved_rating_mapping.pkl")
print("4. improved_domains.pkl")
print("5. improved_training_results.csv")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)