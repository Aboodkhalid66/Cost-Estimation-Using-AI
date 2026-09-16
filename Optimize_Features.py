import pandas as pd
import numpy as np
import os

from scipy.io import arff

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET = os.path.join(
    BASE_DIR,
    "nasa93.arff"
)


print("=" * 75)
print("NASA93 - FEATURE SET OPTIMIZATION")
print("=" * 75)


# ============================================================
# 1. LOAD DATASET
# ============================================================

if not os.path.exists(DATASET):

    print("\nERROR: nasa93.arff not found.")
    print(DATASET)

    input("\nPress Enter to exit...")
    raise SystemExit


data, meta = arff.loadarff(DATASET)

df = pd.DataFrame(data)


# Convert bytes to strings
for column in df.columns:

    if df[column].dtype == object:

        df[column] = df[column].apply(
            lambda x:
            x.decode("utf-8")
            if isinstance(x, bytes)
            else x
        )


print(f"\nProjects: {len(df)}")


# ============================================================
# 2. RATING MAPPING
# ============================================================

rating_mapping = {

    "vl": 0.50,
    "l": 0.75,
    "n": 1.00,
    "h": 1.15,
    "vh": 1.30,
    "xh": 1.45

}


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


for feature in rating_features:

    df[feature] = (

        df[feature]
        .astype(str)
        .str.lower()
        .map(rating_mapping)

    )


# ============================================================
# 3. NUMERIC CONVERSION
# ============================================================

numeric_features = [

    "equivphyskloc",
    "act_effort"

]


for feature in numeric_features:

    df[feature] = pd.to_numeric(
        df[feature],
        errors="coerce"
    )


# ============================================================
# 4. CLEAN DATA
# ============================================================

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

df = df.dropna(
    subset=[
        "equivphyskloc",
        "act_effort"
    ]
)


# ============================================================
# 5. ONE-HOT ENCODE DOMAIN
# ============================================================

df = pd.get_dummies(

    df,

    columns=["cat2"],

    drop_first=True,

    dtype=float

)


# ============================================================
# 6. ALL AVAILABLE FEATURES
# ============================================================

domain_features = [

    column

    for column in df.columns

    if column.startswith("cat2_")

]


# ============================================================
# FEATURE SETS
# ============================================================

feature_sets = {

    # --------------------------------------------------------
    # SET 1
    # Current user-friendly model
    # --------------------------------------------------------

    "Current 9 Inputs":

    [
        "equivphyskloc",
        "cplx",
        "aexp",
        "pcap",
        "lexp",
        "acap",
        "tool",
        "rely"
    ]
    + domain_features,


    # --------------------------------------------------------
    # SET 2
    # Size + all NASA93 effort drivers
    # --------------------------------------------------------

    "All Effort Drivers":

    [
        "equivphyskloc",

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
    + domain_features,


    # --------------------------------------------------------
    # SET 3
    # Size + most important technical drivers
    # --------------------------------------------------------

    "Technical Drivers":

    [
        "equivphyskloc",

        "rely",
        "data",
        "cplx",
        "time",
        "stor",
        "turn",

        "acap",
        "aexp",
        "pcap",
        "lexp",
        "tool"

    ]
    + domain_features,


    # --------------------------------------------------------
    # SET 4
    # Size + selected human factors
    # --------------------------------------------------------

    "Human + Technical":

    [
        "equivphyskloc",

        "cplx",
        "rely",

        "acap",
        "aexp",
        "pcap",
        "lexp",

        "tool",
        "modp",
        "sced"

    ]
    + domain_features

}


# ============================================================
# 7. CROSS VALIDATION
# ============================================================

kf = KFold(

    n_splits=5,

    shuffle=True,

    random_state=42

)


results = []


# ============================================================
# 8. TEST EACH FEATURE SET
# ============================================================

for set_name, features in feature_sets.items():

    print("\n")
    print("=" * 75)
    print(set_name)
    print("=" * 75)


    # Keep only features that actually exist
    available_features = [

        feature

        for feature in features

        if feature in df.columns

    ]


    X = df[
        available_features
    ].copy()

    y = df[
        "act_effort"
    ].copy()


    # Numeric
    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )


    # Missing values
    X = X.fillna(
        X.median()
    )

    X = X.fillna(0)


    # Log target
    y_log = np.log1p(y)


    mae_scores = []

    rmse_scores = []

    r2_scores = []


    for fold, (train_index, test_index) in enumerate(
        kf.split(X),
        start=1
    ):

        X_train = X.iloc[train_index]

        X_test = X.iloc[test_index]

        y_train = y_log.iloc[train_index]

        y_test = y.iloc[test_index]


        # ----------------------------------------------------
        # Random Forest
        # ----------------------------------------------------

        model = RandomForestRegressor(

            n_estimators=300,

            max_depth=8,

            min_samples_leaf=2,

            random_state=42,

            n_jobs=-1

        )


        model.fit(

            X_train,

            y_train

        )


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        predicted_log = model.predict(
            X_test
        )


        predicted_effort = np.expm1(
            predicted_log
        )


        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        mae = mean_absolute_error(

            y_test,

            predicted_effort

        )


        rmse = np.sqrt(

            mean_squared_error(

                y_test,

                predicted_effort

            )

        )


        r2 = r2_score(

            y_test,

            predicted_effort

        )


        mae_scores.append(mae)

        rmse_scores.append(rmse)

        r2_scores.append(r2)


    # ========================================================
    # AVERAGE
    # ========================================================

    mean_mae = np.mean(
        mae_scores
    )

    mean_rmse = np.mean(
        rmse_scores
    )

    mean_r2 = np.mean(
        r2_scores
    )


    results.append({

        "Feature Set": set_name,

        "Features": len(available_features),

        "MAE": mean_mae,

        "RMSE": mean_rmse,

        "R2": mean_r2

    })


    print(
        f"\nFeatures : {len(available_features)}"
    )

    print(
        f"Mean MAE : {mean_mae:.2f}"
    )

    print(
        f"Mean RMSE: {mean_rmse:.2f}"
    )

    print(
        f"Mean R2  : {mean_r2:.4f}"
    )


# ============================================================
# 9. FINAL COMPARISON
# ============================================================

results_df = pd.DataFrame(
    results
)


print("\n")
print("=" * 75)
print("FEATURE SET COMPARISON")
print("=" * 75)


print(

    results_df.to_string(
        index=False
    )

)


# ============================================================
# 10. BEST FEATURE SET
# ============================================================

best = results_df.sort_values(

    by=[
        "MAE",
        "RMSE"
    ],

    ascending=[
        True,
        True
    ]

).iloc[0]


print("\n")
print("=" * 75)
print("BEST FEATURE SET")
print("=" * 75)


print(
    f"\nFeature Set: "
    f"{best['Feature Set']}"
)

print(
    f"Number of Features: "
    f"{int(best['Features'])}"
)

print(
    f"Mean MAE: "
    f"{best['MAE']:.2f}"
)

print(
    f"Mean RMSE: "
    f"{best['RMSE']:.2f}"
)

print(
    f"Mean R2: "
    f"{best['R2']:.4f}"
)


# ============================================================
# 11. SAVE RESULTS
# ============================================================

results_path = os.path.join(

    BASE_DIR,

    "feature_optimization_results.csv"

)


results_df.to_csv(

    results_path,

    index=False

)


print("\n")
print("Results saved to:")

print(results_path)


print("\n")
print("=" * 75)
print("FEATURE OPTIMIZATION COMPLETED")
print("=" * 75)