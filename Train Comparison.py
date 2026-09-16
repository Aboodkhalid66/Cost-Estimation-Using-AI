import os
import joblib
import numpy as np
import pandas as pd

from scipy.io import arff

from sklearn.model_selection import KFold, cross_val_predict
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "nasa93.arff"
)

FEATURES_PATH = os.path.join(
    BASE_DIR,
    "monotonic_model_features.pkl"
)

WEIGHTS_PATH = os.path.join(
    BASE_DIR,
    "monotonic_driver_weights.pkl"
)

DOMAINS_PATH = os.path.join(
    BASE_DIR,
    "monotonic_domains.pkl"
)


# =========================================================
# LOAD DATA
# =========================================================

print("\nLoading NASA93 dataset...")

raw_data, meta = arff.loadarff(DATA_PATH)

df = pd.DataFrame(raw_data)


# Decode byte columns
for column in df.columns:

    if df[column].dtype == object:

        df[column] = df[column].apply(

            lambda x:
            x.decode("utf-8")
            if isinstance(x, bytes)
            else x

        )


print("Dataset loaded successfully.")
print("Rows:", len(df))
print("Columns:", len(df.columns))


# =========================================================
# DRIVERS
# =========================================================

drivers = [
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


# =========================================================
# NASA93 RATING VALUES
# =========================================================

rating_values = {
    "vl": 0.5,
    "l": 0.75,
    "n": 1.0,
    "h": 1.15,
    "vh": 1.30,
    "xh": 1.45
}


# =========================================================
# LOAD FILES FROM CURRENT MODEL
# =========================================================

model_features = joblib.load(
    FEATURES_PATH
)

driver_weights = joblib.load(
    WEIGHTS_PATH
)

domains = joblib.load(
    DOMAINS_PATH
)


print("\nLoaded model features:", len(model_features))
print("Loaded domains:", len(domains))


# =========================================================
# CLEAN DOMAIN
# =========================================================

df["cat2"] = (
    df["cat2"]
    .astype(str)
    .str.strip()
)


# =========================================================
# DRIVER ENCODING
# =========================================================

for driver in drivers:

    df[driver] = (
        df[driver]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df[driver] = df[driver].map(
        rating_values
    )

    df[driver] = df[driver].fillna(1.0)


# =========================================================
# KLOC
# =========================================================

df["equivphyskloc"] = pd.to_numeric(
    df["equivphyskloc"],
    errors="coerce"
)

df["equivphyskloc"] = (
    df["equivphyskloc"]
    .fillna(
        df["equivphyskloc"].median()
    )
)

df["log_equivphyskloc"] = np.log1p(
    df["equivphyskloc"]
)


# =========================================================
# WEIGHT FEATURES
# =========================================================

print("\nCreating driver weight features...")


for driver in drivers:

    weight_column = driver + "_weight"

    weights = []


    for value in df[driver]:

        rating = "n"

        # Find closest rating value
        closest_rating = min(
            rating_values,
            key=lambda r:
            abs(
                rating_values[r] - value
            )
        )

        rating = closest_rating


        weight = value


        try:

            if driver in driver_weights:

                mapping = driver_weights[driver]

                if isinstance(mapping, dict):

                    if rating in mapping:

                        weight = mapping[rating]

        except Exception:

            pass


        weights.append(weight)


    df[weight_column] = weights


# =========================================================
# DOMAIN ONE-HOT ENCODING
# =========================================================

print("\nEncoding project domains...")


domain_dummies = pd.get_dummies(
    df["cat2"],
    prefix="cat2",
    drop_first=True
)


# =========================================================
# BUILD FEATURE DATAFRAME
# =========================================================

X = pd.DataFrame(index=df.index)


# KLOC
X["equivphyskloc"] = (
    df["equivphyskloc"]
)


# Log KLOC
X["log_equivphyskloc"] = (
    df["log_equivphyskloc"]
)


# Drivers
for driver in drivers:

    X[driver] = df[driver]


# Driver weights
for driver in drivers:

    weight_column = driver + "_weight"

    X[weight_column] = (
        df[weight_column]
    )


# Domain features
for column in domain_dummies.columns:

    X[column] = domain_dummies[column]


# =========================================================
# MATCH EXACT MODEL FEATURE ORDER
# =========================================================

X = X.reindex(
    columns=model_features,
    fill_value=0
)


# =========================================================
# TARGET
# =========================================================

y = pd.to_numeric(
    df["act_effort"],
    errors="coerce"
)

valid = y.notna()

X = X.loc[valid].copy()

y = y.loc[valid].copy()


# Log target
y_log = np.log1p(y)


print("\nFinal training shape:")
print("X:", X.shape)
print("y:", y.shape)


# =========================================================
# CROSS VALIDATION
# =========================================================

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# =========================================================
# DECISION TREE
# =========================================================

print("\n==============================")
print("DECISION TREE")
print("==============================")


decision_tree = DecisionTreeRegressor(

    max_depth=8,

    min_samples_leaf=2,

    random_state=42

)


dt_predictions_log = cross_val_predict(

    decision_tree,

    X,

    y_log,

    cv=kf

)


dt_predictions = np.expm1(
    dt_predictions_log
)


dt_mae = mean_absolute_error(
    y,
    dt_predictions
)


dt_rmse = np.sqrt(
    mean_squared_error(
        y,
        dt_predictions
    )
)


dt_r2 = r2_score(
    y,
    dt_predictions
)


print(
    f"MAE  : {dt_mae:.2f}"
)

print(
    f"RMSE : {dt_rmse:.2f}"
)

print(
    f"R2   : {dt_r2:.4f}"
)


# =========================================================
# RANDOM FOREST
# =========================================================

print("\n==============================")
print("RANDOM FOREST")
print("==============================")


random_forest = RandomForestRegressor(

    n_estimators=300,

    max_depth=8,

    min_samples_leaf=2,

    random_state=42,

    n_jobs=-1

)


rf_predictions_log = cross_val_predict(

    random_forest,

    X,

    y_log,

    cv=kf

)


rf_predictions = np.expm1(
    rf_predictions_log
)


rf_mae = mean_absolute_error(
    y,
    rf_predictions
)


rf_rmse = np.sqrt(
    mean_squared_error(
        y,
        rf_predictions
    )
)


rf_r2 = r2_score(
    y,
    rf_predictions
)


print(
    f"MAE  : {rf_mae:.2f}"
)

print(
    f"RMSE : {rf_rmse:.2f}"
)

print(
    f"R2   : {rf_r2:.4f}"
)


# =========================================================
# TRAIN FINAL MODELS ON FULL DATA
# =========================================================

print("\nTraining final models...")


decision_tree.fit(
    X,
    y_log
)


random_forest.fit(
    X,
    y_log
)


# =========================================================
# SAVE MODELS
# =========================================================

joblib.dump(

    decision_tree,

    os.path.join(
        BASE_DIR,
        "decision_tree_effort_model.pkl"
    )

)


joblib.dump(

    random_forest,

    os.path.join(
        BASE_DIR,
        "random_forest_comparison_model.pkl"
    )

)


# =========================================================
# SAVE RESULTS
# =========================================================

results = pd.DataFrame({

    "Model": [
        "Decision Tree",
        "Random Forest"
    ],

    "MAE": [
        dt_mae,
        rf_mae
    ],

    "RMSE": [
        dt_rmse,
        rf_rmse
    ],

    "R2": [
        dt_r2,
        rf_r2
    ]

})


results.to_csv(

    os.path.join(
        BASE_DIR,
        "algorithm_comparison_results.csv"
    ),

    index=False

)


# =========================================================
# PRINT RESULTS
# =========================================================

print("\n")
print("==========================================")
print("ALGORITHM COMPARISON COMPLETED")
print("==========================================")


print("\nDecision Tree:")
print(
    f"MAE  = {dt_mae:.2f}"
)
print(
    f"RMSE = {dt_rmse:.2f}"
)
print(
    f"R2   = {dt_r2:.4f}"
)


print("\nRandom Forest:")
print(
    f"MAE  = {rf_mae:.2f}"
)
print(
    f"RMSE = {rf_rmse:.2f}"
)
print(
    f"R2   = {rf_r2:.4f}"
)


print("\nCreated files:")

print(
    "decision_tree_effort_model.pkl"
)

print(
    "random_forest_comparison_model.pkl"
)

print(
    "algorithm_comparison_results.csv"
)

print("\n==========================================")