from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)


# =========================================================
# LOAD MAIN MODEL
# =========================================================

MODEL_FILE = "monotonic_effort_model.pkl"
FEATURES_FILE = "monotonic_model_features.pkl"
WEIGHTS_FILE = "monotonic_driver_weights.pkl"
DOMAINS_FILE = "monotonic_domains.pkl"

model = joblib.load(MODEL_FILE)
model_features = joblib.load(FEATURES_FILE)
driver_weights = joblib.load(WEIGHTS_FILE)
domains = joblib.load(DOMAINS_FILE)


# =========================================================
# COMPARISON FILE
# =========================================================

COMPARISON_FILE = "algorithm_comparison_results.csv"


# =========================================================
# PROJECT TYPES
# =========================================================

PROJECT_TYPES = {
    "store": "utility",
    "hospital": "realdataprocessing",
    "bank": "datacapture",
    "university": "application_ground",
    "mobile": "application_ground",
    "communication": "communications",
    "management": "monitor_control",
    "simulation": "simulation",
    "operating_system": "operatingsystem",
    "scientific": "science"
}


PROJECT_TYPE_NAMES_AR = {
    "store": "متجر إلكتروني",
    "hospital": "نظام مستشفى",
    "bank": "نظام بنكي",
    "university": "نظام جامعة",
    "mobile": "تطبيق موبايل",
    "communication": "نظام اتصالات",
    "management": "نظام إدارة",
    "simulation": "نظام محاكاة",
    "operating_system": "نظام تشغيل",
    "scientific": "نظام علمي"
}


PROJECT_TYPE_NAMES_EN = {
    "store": "E-Commerce Store",
    "hospital": "Hospital System",
    "bank": "Banking System",
    "university": "University System",
    "mobile": "Mobile Application",
    "communication": "Communication System",
    "management": "Management System",
    "simulation": "Simulation System",
    "operating_system": "Operating System",
    "scientific": "Scientific System"
}


# =========================================================
# IMPORTANT:
# index.html EXPECTS project_types AS OBJECTS
# =========================================================

project_types = []

for key in PROJECT_TYPES:

    project_types.append({
        "id": key,
        "name_ar": PROJECT_TYPE_NAMES_AR[key],
        "name_en": PROJECT_TYPE_NAMES_EN[key]
    })


# =========================================================
# RATING VALUES
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
# RATING NAMES
# =========================================================

rating_names_ar = {
    "vl": "منخفض جداً",
    "l": "منخفض",
    "n": "متوسط",
    "h": "مرتفع",
    "vh": "مرتفع جداً",
    "xh": "مرتفع جداً جداً"
}


rating_names_en = {
    "vl": "Very Low",
    "l": "Low",
    "n": "Nominal",
    "h": "High",
    "vh": "Very High",
    "xh": "Extra High"
}


# =========================================================
# EFFORT DRIVERS
# =========================================================

drivers = {

    "rely": {
        "ar": "موثوقية النظام",
        "en": "Required Software Reliability",
        "question_ar": "ما مدى أهمية أن يعمل النظام بدون أخطاء؟",
        "question_en": "How important is it for the system to operate without failures?"
    },

    "data": {
        "ar": "حجم البيانات",
        "en": "Database Size",
        "question_ar": "ما حجم البيانات التي سيتعامل معها النظام؟",
        "question_en": "How large is the amount of data handled by the system?"
    },

    "cplx": {
        "ar": "تعقيد النظام",
        "en": "Product Complexity",
        "question_ar": "ما مدى تعقيد العمليات والوظائف داخل النظام؟",
        "question_en": "How complex are the operations and functions of the system?"
    },

    "time": {
        "ar": "قيود وقت التنفيذ",
        "en": "Execution Time Constraint",
        "question_ar": "هل توجد قيود صارمة على سرعة تنفيذ النظام؟",
        "question_en": "Are there strict constraints on system execution time?"
    },

    "stor": {
        "ar": "قيود الذاكرة والتخزين",
        "en": "Main Storage Constraint",
        "question_ar": "هل توجد قيود على الذاكرة أو مساحة التخزين؟",
        "question_en": "Are there constraints on memory or storage?"
    },

    "virt": {
        "ar": "تغير بيئة التشغيل",
        "en": "Virtual Machine Volatility",
        "question_ar": "ما مدى تغير بيئة التشغيل أو الأجهزة المستخدمة؟",
        "question_en": "How frequently does the operating environment change?"
    },

    "turn": {
        "ar": "زمن الاستجابة",
        "en": "Computer Turnaround Time",
        "question_ar": "ما مدى أهمية سرعة استجابة النظام؟",
        "question_en": "How important is fast system response?"
    },

    "acap": {
        "ar": "قدرة محللي النظام",
        "en": "Analyst Capability",
        "question_ar": "ما مستوى خبرة وقدرة محللي النظام؟",
        "question_en": "What is the capability level of the system analysts?"
    },

    "aexp": {
        "ar": "خبرة المحللين",
        "en": "Applications Experience",
        "question_ar": "ما مستوى خبرة الفريق في مجال التطبيقات؟",
        "question_en": "What is the team's experience with similar applications?"
    },

    "pcap": {
        "ar": "قدرة المبرمجين",
        "en": "Programmer Capability",
        "question_ar": "ما مستوى خبرة وقدرة المبرمجين؟",
        "question_en": "What is the capability level of the programmers?"
    },

    "vexp": {
        "ar": "خبرة المنصة",
        "en": "Virtual Machine Experience",
        "question_ar": "ما مستوى خبرة الفريق ببيئة التشغيل والمنصة؟",
        "question_en": "What is the team's experience with the platform?"
    },

    "lexp": {
        "ar": "خبرة لغة البرمجة",
        "en": "Programming Language Experience",
        "question_ar": "ما مستوى خبرة الفريق بلغة البرمجة المستخدمة؟",
        "question_en": "What is the team's experience with the programming language?"
    },

    "modp": {
        "ar": "استخدام ممارسات حديثة",
        "en": "Modern Programming Practices",
        "question_ar": "إلى أي مدى يستخدم الفريق ممارسات برمجية حديثة؟",
        "question_en": "To what extent does the team use modern programming practices?"
    },

    "tool": {
        "ar": "استخدام أدوات البرمجة",
        "en": "Software Tools Usage",
        "question_ar": "ما مدى استخدام أدوات وبرامج تساعد في التطوير؟",
        "question_en": "How extensively are software development tools used?"
    },

    "sced": {
        "ar": "ضغط جدول التسليم",
        "en": "Required Development Schedule",
        "question_ar": "ما مدى ضغط الوقت المحدد لتسليم المشروع؟",
        "question_en": "How tight is the required development schedule?"
    }
}


# =========================================================
# ADD RATING OPTIONS TO EACH DRIVER
# =========================================================

for driver_name in drivers:

    drivers[driver_name]["values"] = []

    for code in rating_names_ar:

        drivers[driver_name]["values"].append({
            "code": code,
            "name_ar": rating_names_ar[code],
            "name_en": rating_names_en[code]
        })



# =========================================================
# DRIVER WEIGHTS
# =========================================================

default_driver_weights = {

    "rely": {
        "vl": 0.75,
        "l": 0.88,
        "n": 1.00,
        "h": 1.15,
        "vh": 1.40,
        "xh": 1.40
    },

    "data": {
        "vl": 0.93,
        "l": 0.93,
        "n": 1.00,
        "h": 1.08,
        "vh": 1.16,
        "xh": 1.16
    },

    "cplx": {
        "vl": 0.70,
        "l": 0.85,
        "n": 1.00,
        "h": 1.15,
        "vh": 1.30,
        "xh": 1.65
    },

    "time": {
        "vl": 1.00,
        "l": 1.00,
        "n": 1.00,
        "h": 1.11,
        "vh": 1.30,
        "xh": 1.66
    },

    "stor": {
        "vl": 1.00,
        "l": 1.00,
        "n": 1.00,
        "h": 1.06,
        "vh": 1.21,
        "xh": 1.56
    },

    "virt": {
        "vl": 0.87,
        "l": 0.87,
        "n": 1.00,
        "h": 1.15,
        "vh": 1.30,
        "xh": 1.30
    },

    "turn": {
        "vl": 0.87,
        "l": 0.87,
        "n": 1.00,
        "h": 1.07,
        "vh": 1.15,
        "xh": 1.30
    },

    "acap": {
        "vl": 1.42,
        "l": 1.19,
        "n": 1.00,
        "h": 0.85,
        "vh": 0.71,
        "xh": 0.71
    },

    "aexp": {
        "vl": 1.22,
        "l": 1.10,
        "n": 1.00,
        "h": 0.88,
        "vh": 0.81,
        "xh": 0.81
    },

    "pcap": {
        "vl": 1.34,
        "l": 1.15,
        "n": 1.00,
        "h": 0.88,
        "vh": 0.76,
        "xh": 0.76
    },

    "vexp": {
        "vl": 1.21,
        "l": 1.10,
        "n": 1.00,
        "h": 0.90,
        "vh": 0.90,
        "xh": 0.90
    },

    "lexp": {
        "vl": 1.14,
        "l": 1.07,
        "n": 1.00,
        "h": 0.95,
        "vh": 0.95,
        "xh": 0.95
    },

    "modp": {
        "vl": 1.24,
        "l": 1.10,
        "n": 1.00,
        "h": 0.91,
        "vh": 0.82,
        "xh": 0.82
    },

    "tool": {
        "vl": 1.24,
        "l": 1.10,
        "n": 1.00,
        "h": 0.91,
        "vh": 0.82,
        "xh": 0.82
    },

    "sced": {
        "vl": 1.23,
        "l": 1.08,
        "n": 1.00,
        "h": 1.04,
        "vh": 1.10,
        "xh": 1.10
    }
}


# =========================================================
# COMPLETE MISSING WEIGHTS IF NEEDED
# =========================================================

if isinstance(driver_weights, dict):

    for driver_name in default_driver_weights:

        if driver_name not in driver_weights:

            driver_weights[driver_name] = (
                default_driver_weights[driver_name]
            )


# =========================================================
# CURRENCY RATES
# =========================================================

currency_rates = {
    "USD": 1.0,
    "SAR": 3.75,
    "AED": 3.65,
    "EGP": 50.0
}


# =========================================================
# BUILD MODEL FEATURES
# =========================================================

def build_features(
    project_type,
    kloc,
    selected_ratings
):

    internal_domain = PROJECT_TYPES.get(
        project_type,
        "utility"
    )

    data = {}

    # -----------------------------------------------------
    # PROJECT SIZE
    # -----------------------------------------------------

    kloc = float(kloc)

    data["equivphyskloc"] = kloc

    data["log_equivphyskloc"] = np.log1p(
        kloc
    )

    # -----------------------------------------------------
    # EFFORT DRIVERS
    # -----------------------------------------------------

    for driver_name in drivers:

        rating = selected_ratings.get(
            driver_name,
            "n"
        )

        if rating not in rating_values:

            rating = "n"

        # Original numeric driver value
        data[driver_name] = rating_values[rating]

        # COCOMO-style weight
        weights = driver_weights.get(
            driver_name,
            default_driver_weights[driver_name]
        )

        data[f"{driver_name}_weight"] = weights.get(
            rating,
            1.0
        )

    # -----------------------------------------------------
    # PROJECT DOMAIN ONE-HOT
    # -----------------------------------------------------

    for domain in domains:

        column_name = str(domain)

        if column_name.lower() == internal_domain.lower():

            data[column_name] = 1

        else:

            data[column_name] = 0

    # -----------------------------------------------------
    # DATAFRAME
    # -----------------------------------------------------

    X = pd.DataFrame(
        [data]
    )

    # -----------------------------------------------------
    # EXACT MODEL FEATURES
    # -----------------------------------------------------

    X = X.reindex(
        columns=model_features,
        fill_value=0
    )

    X = X.fillna(0)

    return X


# =========================================================
# WHAT-IF RECOMMENDATION ENGINE
# =========================================================

def predict_effort_and_cost(project_type, kloc, selected_ratings, cost_per_pm):
    """Run the same production model used by the estimator."""
    X = build_features(project_type, kloc, selected_ratings)
    predicted_log_effort = float(model.predict(X)[0])
    effort = max(0.0, float(np.expm1(predicted_log_effort)))
    cost = effort * float(cost_per_pm)
    return effort, cost


def generate_recommendations(project_type, kloc, selected_ratings, cost_per_pm, max_items=5):
    """Generate model-based What-If suggestions by changing one driver at a time.

    These are simulations, not causal guarantees. Size is intentionally not changed
    because increasing KLOC generally increases effort/cost rather than improving it.
    """
    rating_order = ["vl", "l", "n", "h", "vh", "xh"]
    baseline_effort, baseline_cost = predict_effort_and_cost(
        project_type, kloc, selected_ratings, cost_per_pm
    )

    candidates = []

    # Driver direction based on the COCOMO-style weights used by the model.
    # For product/constraint drivers, lowering the requirement can reduce effort,
    # but the UI must make clear that this is only appropriate when requirements allow.
    product_drivers = {"rely", "data", "cplx", "time", "stor", "virt", "turn", "sced"}
    team_drivers = {"acap", "aexp", "pcap", "vexp", "lexp", "modp", "tool"}

    for driver_name in drivers:
        current = selected_ratings.get(driver_name, "n")
        if current not in rating_order:
            current = "n"
        idx = rating_order.index(current)

        target_indices = []
        direction_text = ""
        condition_text_ar = ""
        condition_text_en = ""

        if driver_name in team_drivers:
            # Increasing team capability/experience/tooling is the practical
            # improvement direction in the model.
            if idx < len(rating_order) - 1:
                target_indices.append(idx + 1)
                direction_text = "increase"
                condition_text_ar = "إذا كان رفع مستوى الخبرة أو الأدوات متاحاً"
                condition_text_en = "if improving team capability or tooling is feasible"
        elif driver_name in product_drivers:
            # Lowering a requirement can reduce effort, but only when acceptable.
            if idx > 0:
                target_indices.append(idx - 1)
                direction_text = "decrease"
                condition_text_ar = "إذا كان تخفيف هذا المتطلب مقبولاً وظيفياً"
                condition_text_en = "if reducing this requirement is functionally acceptable"

        for target_idx in target_indices:
            target = rating_order[target_idx]
            trial = dict(selected_ratings)
            trial[driver_name] = target
            trial_effort, trial_cost = predict_effort_and_cost(
                project_type, kloc, trial, cost_per_pm
            )

            effort_change = trial_effort - baseline_effort
            cost_change = trial_cost - baseline_cost
            if cost_change >= -1e-9:
                continue

            savings = abs(cost_change)
            savings_pct = (savings / baseline_cost * 100.0) if baseline_cost > 0 else 0.0
            effort_reduction = max(0.0, -effort_change)
            effort_pct = (effort_reduction / baseline_effort * 100.0) if baseline_effort > 0 else 0.0

            candidates.append({
                "driver": driver_name,
                "driver_ar": drivers[driver_name]["ar"],
                "driver_en": drivers[driver_name]["en"],
                "current_ar": rating_names_ar.get(current, current),
                "current_en": rating_names_en.get(current, current),
                "target_ar": rating_names_ar.get(target, target),
                "target_en": rating_names_en.get(target, target),
                "direction": direction_text,
                "condition_ar": condition_text_ar,
                "condition_en": condition_text_en,
                "effort": round(trial_effort, 2),
                "cost": round(trial_cost, 2),
                "effort_reduction": round(effort_reduction, 2),
                "effort_pct": round(effort_pct, 2),
                "cost_saving": round(savings, 2),
                "cost_saving_pct": round(savings_pct, 2),
            })

    candidates.sort(key=lambda x: (x["cost_saving"], x["effort_reduction"]), reverse=True)

    return {
        "baseline_effort": round(baseline_effort, 2),
        "baseline_cost": round(baseline_cost, 2),
        "items": candidates[:max_items]
    }


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def index():

    return render_template(
        "index.html",

        # Drivers
        drivers=drivers,
        driver_info=drivers,

        # Project types
        project_types=project_types,
        project_types_ar=PROJECT_TYPE_NAMES_AR,
        project_types_en=PROJECT_TYPE_NAMES_EN,

        # Ratings
        rating_names_ar=rating_names_ar,
        rating_names_en=rating_names_en
    )


# =========================================================
# PREDICTION
# =========================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        # -------------------------------------------------
        # PROJECT TYPE
        # -------------------------------------------------

        project_type = request.form.get(
            "project_type",
            "store"
        )

        if project_type not in PROJECT_TYPES:

            project_type = "store"

        # -------------------------------------------------
        # LINES OF CODE
        # -------------------------------------------------

        lines_raw = request.form.get("lines_of_code")

        # Support both the current HTML field (lines_of_code)
        # and KLOC if an older HTML form is used.
        if lines_raw is None or str(lines_raw).strip() == "":
            kloc_raw = request.form.get("kloc")
            if kloc_raw is not None and str(kloc_raw).strip() != "":
                lines_of_code = float(kloc_raw) * 1000.0
            else:
                lines_of_code = 0.0
        else:
            lines_of_code = float(lines_raw)

        if lines_of_code < 0:

            lines_of_code = 0

        # Convert Lines → KLOC
        kloc = lines_of_code / 1000.0

        # -------------------------------------------------
        # COST PER PERSON-MONTH
        # -------------------------------------------------

        cost_per_pm = float(
            request.form.get(
                "cost_per_pm",
                0
            )
        )

        if cost_per_pm < 0:

            cost_per_pm = 0

        # -------------------------------------------------
        # GET ALL DRIVER RATINGS
        # -------------------------------------------------

        selected_ratings = {}

        for driver_name in drivers:

            selected_ratings[driver_name] = request.form.get(
                driver_name,
                "n"
            )

        # -------------------------------------------------
        # BUILD MODEL INPUT
        # -------------------------------------------------

        X = build_features(
            project_type,
            kloc,
            selected_ratings
        )

        # -------------------------------------------------
        # MODEL PREDICTION
        # -------------------------------------------------

        predicted_log_effort = model.predict(
            X
        )[0]

        # Convert log effort → actual effort
        estimated_effort = np.expm1(
            predicted_log_effort
        )

        estimated_effort = max(
            0,
            float(estimated_effort)
        )

        # -------------------------------------------------
        # COST CALCULATION
        # -------------------------------------------------

        estimated_cost_usd = (
            estimated_effort *
            cost_per_pm
        )

        estimated_cost_sar = (
            estimated_cost_usd *
            currency_rates["SAR"]
        )

        estimated_cost_aed = (
            estimated_cost_usd *
            currency_rates["AED"]
        )

        estimated_cost_egp = (
            estimated_cost_usd *
            currency_rates["EGP"]
        )

        # -------------------------------------------------
        # WHAT-IF RECOMMENDATIONS
        # -------------------------------------------------
        recommendations = generate_recommendations(
            project_type,
            kloc,
            selected_ratings,
            cost_per_pm
        )

        # -------------------------------------------------
        # PROJECT NAME
        # -------------------------------------------------

        project_name_ar = PROJECT_TYPE_NAMES_AR.get(
            project_type,
            project_type
        )

        project_name_en = PROJECT_TYPE_NAMES_EN.get(
            project_type,
            project_type
        )

        # -------------------------------------------------
        # SELECTED PROJECT
        # -------------------------------------------------

        selected_project = {

            "id": project_type,

            "name_ar": project_name_ar,

            "name_en": project_name_en
        }

        # -------------------------------------------------
        # INPUT TABLE
        # -------------------------------------------------

        input_rows = []

        # Project type
        input_rows.append({

            "name_ar": "نوع المشروع",

            "name_en": "Project Type",

            "value_ar": project_name_ar,

            "value_en": project_name_en
        })

        # Lines
        input_rows.append({

            "name_ar": "حجم المشروع",

            "name_en": "Project Size",

            "value_ar": (
                f"{lines_of_code:,.0f} سطر"
            ),

            "value_en": (
                f"{lines_of_code:,.0f} Lines"
            )
        })

        # KLOC
        input_rows.append({

            "name_ar": "حجم المشروع KLOC",

            "name_en": "Project Size KLOC",

            "value_ar": f"{kloc:.2f}",

            "value_en": f"{kloc:.2f}"
        })

        # Cost per PM
        input_rows.append({

            "name_ar": "تكلفة الشخص/شهر",

            "name_en": "Cost per Person-Month",

            "value_ar": f"${cost_per_pm:,.2f}",

            "value_en": f"${cost_per_pm:,.2f}"
        })

        # Drivers
        for driver_name in drivers:

            rating = selected_ratings.get(
                driver_name,
                "n"
            )

            input_rows.append({

                "name_ar": drivers[driver_name]["ar"],

                "name_en": drivers[driver_name]["en"],

                "value_ar": rating_names_ar.get(
                    rating,
                    rating
                ),

                "value_en": rating_names_en.get(
                    rating,
                    rating
                )
            })

        # -------------------------------------------------
        # RESULT OBJECT
        # -------------------------------------------------

        result = {

            "estimated_effort": round(
                estimated_effort,
                2
            ),

            "estimated_cost_usd": round(
                estimated_cost_usd,
                2
            ),

            "estimated_cost_sar": round(
                estimated_cost_sar,
                2
            ),

            "estimated_cost_aed": round(
                estimated_cost_aed,
                2
            ),

            "estimated_cost_egp": round(
                estimated_cost_egp,
                2
            )
        }

        # -------------------------------------------------
        # RENDER MAIN PAGE
        # -------------------------------------------------

        return render_template(

            "index.html",

            drivers=drivers,

            driver_info=drivers,

            project_types=project_types,

            project_types_ar=PROJECT_TYPE_NAMES_AR,

            project_types_en=PROJECT_TYPE_NAMES_EN,

            rating_names_ar=rating_names_ar,

            rating_names_en=rating_names_en,

            result=result,

            input_rows=input_rows,

            selected_project=selected_project,

            selected_project_type=project_type,

            selected_ratings=selected_ratings,

            lines_of_code=lines_of_code,

            kloc=kloc,

            cost_per_pm=cost_per_pm,

            estimated_effort=result[
                "estimated_effort"
            ],

            estimated_cost_usd=result[
                "estimated_cost_usd"
            ],

            estimated_cost_sar=result[
                "estimated_cost_sar"
            ],

            estimated_cost_aed=result[
                "estimated_cost_aed"
            ],

            estimated_cost_egp=result[
                "estimated_cost_egp"
            ],
            recommendations=recommendations
        )

    except Exception as e:

        # -------------------------------------------------
        # ERROR PAGE
        # -------------------------------------------------

        return render_template(

            "index.html",

            drivers=drivers,

            driver_info=drivers,

            project_types=project_types,

            project_types_ar=PROJECT_TYPE_NAMES_AR,

            project_types_en=PROJECT_TYPE_NAMES_EN,

            rating_names_ar=rating_names_ar,

            rating_names_en=rating_names_en,

            error=str(e)
        )


# =========================================================
# ALGORITHM COMPARISON PAGE
# =========================================================

@app.route("/comparison")
def comparison():

    try:

        # -------------------------------------------------
        # CHECK CSV
        # -------------------------------------------------

        if not os.path.exists(
            COMPARISON_FILE
        ):

            return """
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <title>Algorithm Comparison</title>
            </head>

            <body
                style="
                    font-family:Arial;
                    padding:40px;
                    direction:ltr;
                "
            >

                <h2>
                    Algorithm comparison file not found.
                </h2>

                <p>
                    Please run:
                </p>

                <pre>
python "Train Comparison.py"
                </pre>

                <br>

                <a href="/">
                    Back to Cost Estimation
                </a>

            </body>
            </html>
            """

        # -------------------------------------------------
        # READ CSV
        # -------------------------------------------------

        df = pd.read_csv(
            COMPARISON_FILE
        )

        # -------------------------------------------------
        # NORMALIZE COLUMN NAMES
        # -------------------------------------------------

        df.columns = [

            str(column).strip()

            for column in df.columns
        ]

        # -------------------------------------------------
        # FIND MODEL COLUMN
        # -------------------------------------------------

        model_column = None

        possible_model_columns = [

            "Model",

            "model",

            "Algorithm",

            "algorithm"
        ]

        for column in possible_model_columns:

            if column in df.columns:

                model_column = column

                break

        if model_column is None:

            model_column = df.columns[0]

        # -------------------------------------------------
        # CREATE RESULT LIST
        # -------------------------------------------------

        comparison_data = []

        for _, row in df.iterrows():

            model_name = str(
                row[model_column]
            )

            mae = float(
                row["MAE"]
            )

            rmse = float(
                row["RMSE"]
            )

            r2 = float(
                row["R2"]
            )

            comparison_data.append({

                "model": model_name,

                "mae": mae,

                "rmse": rmse,

                "r2": r2
            })

        # -------------------------------------------------
        # RENDER COMPARISON PAGE
        # -------------------------------------------------

        return render_template(

            "comparison.html",

            results=comparison_data
        )

    except Exception as e:

        return f"""
        <!DOCTYPE html>

        <html>

        <head>

            <meta charset="UTF-8">

            <title>
                Algorithm Comparison Error
            </title>

        </head>

        <body
            style="
                font-family:Arial;
                padding:40px;
            "
        >

            <h2>
                Error Loading Algorithm Comparison
            </h2>

            <pre>
{str(e)}
            </pre>

            <br>

            <a href="/">
                Back to Cost Estimation
            </a>

        </body>

        </html>
        """


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )