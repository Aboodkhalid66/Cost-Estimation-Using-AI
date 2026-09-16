import pandas as pd
import numpy as np
import os

from scipy.io import arff

from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
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


# ============================================================
# TITLE
# ============================================================

print("=" * 70)
print("SOFTWARE PROJECT EFFORT ESTIMATION")
print("MODEL COMPARISON")
print("=" * 70)


# ============================================================
# 1. CHECK DATASET
# ============================================================

if not os.path.exists(DATASET):

    print("\nERROR: nasa93.arff not found.")

    print("\nExpected location:")
    print(DATASET)

    input("\nPress Enter to exit...")
    raise SystemExit


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\nLoading NASA93 dataset...")

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


print(
    f"Projects found: {len(df)}"
)


# ============================================================
# 3. SELECT FEATURES
# ============================================================

selected_features = [

    "cat2",
    "equivphyskloc",
    "cplx",
    "aexp",
    "pcap",
    "lexp",
    "acap",
    "tool",
    "rely"

]

target = "act_effort"


df = df[
    selected_features + [target]
].copy()


# ============================================================
# 4. RATING MAPPING
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

    "cplx",
    "aexp",
    "pcap",
    "lexp",
    "acap",
    "tool",
    "rely"

]


for feature in rating_features:

    df[feature] = (

        df[feature]
        .astype(str)
        .str.lower()
        .map(rating_mapping)

    )


# ============================================================
# 5. NUMERIC CONVERSION
# ============================================================

df["equivphyskloc"] = pd.to_numeric(
    df["equivphyskloc"],
    errors="coerce"
)

df[target] = pd.to_numeric(
    df[target],
    errors="coerce"
)


# ============================================================
# 6. CLEAN DATA
# ============================================================

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

df = df.dropna()


print(
    f"Valid projects: {len(df)}"
)


# ============================================================
# 7. ONE-HOT ENCODING
# ============================================================

df = pd.get_dummies(

    df,

    columns=["cat2"],

    drop_first=True,

    dtype=float

)


# ============================================================
# 8. PREPARE DATA
# ============================================================

X = df.drop(
    columns=[target]
)

y = df[target]


X = X.apply(
    pd.to_numeric,
    errors="coerce"
)

X = X.fillna(
    X.median()
)

X = X.fillna(0)


# ============================================================
# 9. LOG TARGET
# ============================================================

y_log = np.log1p(y)


# ============================================================
# 10. MODELS
# ============================================================

models = {

    "Linear Regression":
        LinearRegression(),

    "Decision Tree":
        DecisionTreeRegressor(
            max_depth=8,
            min_samples_leaf=2,
            random_state=42
        ),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

}


# ============================================================
# 11. CROSS VALIDATION
# ============================================================

kf = KFold(

    n_splits=5,

    shuffle=True,

    random_state=42

)


results = []


print("\n")
print("=" * 70)
print("5-FOLD CROSS VALIDATION")
print("=" * 70)


for model_name, model in models.items():

    print("\n")
    print("-" * 70)
    print(model_name)
    print("-" * 70)


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


        # Train
        model.fit(
            X_train,
            y_train
        )


        # Predict
        predicted_log = model.predict(
            X_test
        )


        predicted_effort = np.expm1(
            predicted_log
        )


        # Metrics
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


        print(
            f"Fold {fold}: "
            f"MAE={mae:.2f} | "
            f"RMSE={rmse:.2f} | "
            f"R2={r2:.4f}"
        )


    # Mean
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

        "Model": model_name,

        "Mean MAE": mean_mae,

        "Mean RMSE": mean_rmse,

        "Mean R2": mean_r2

    })


    print("\nAverage:")

    print(
        f"MAE  : {mean_mae:.2f}"
    )

    print(
        f"RMSE : {mean_rmse:.2f}"
    )

    print(
        f"R2   : {mean_r2:.4f}"
    )


# ============================================================
# 12. FINAL COMPARISON
# ============================================================

results_df = pd.DataFrame(
    results
)


print("\n")
print("=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)


print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# 13. FIND BEST MODEL
# ============================================================

# الأفضل:
# أقل MAE
# وإذا تقارب MAE نستخدم R2 الأعلى

best_model = results_df.sort_values(

    by=[
        "Mean MAE",
        "Mean RMSE"
    ],

    ascending=[
        True,
        True
    ]

).iloc[0]


print("\n")
print("=" * 70)
print("BEST MODEL")
print("=" * 70)


print(
    f"\nModel: {best_model['Model']}"
)

print(
    f"Mean MAE: "
    f"{best_model['Mean MAE']:.2f}"
)

print(
    f"Mean RMSE: "
    f"{best_model['Mean RMSE']:.2f}"
)

print(
    f"Mean R2: "
    f"{best_model['Mean R2']:.4f}"
)


# ============================================================
# 14. SAVE RESULTS
# ============================================================

results_path = os.path.join(

    BASE_DIR,

    "model_comparison_results.csv"

)


results_df.to_csv(

    results_path,

    index=False

)


print("\n")
print(
    "Results saved to:"
)

print(
    results_path
)


print("\n")
print("=" * 70)
print("COMPARISON COMPLETED")
print("=" * 70)