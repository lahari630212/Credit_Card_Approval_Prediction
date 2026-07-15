from flask import Flask, render_template, request
import pandas as pd
import joblib


app = Flask(__name__)


# ==========================================
# LOAD MODEL AND FEATURE COLUMNS
# ==========================================

model = joblib.load(
    "credit_card_model.pkl"
)

feature_columns = joblib.load(
    "feature_columns.pkl"
)


# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")
def home():

    return render_template(
        "home.html"
    )


# ==========================================
# PREDICTION PAGE
# ==========================================

@app.route("/prediction")
def prediction_page():

    return render_template(
        "index.html"
    )


# ==========================================
# PREDICT
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    # ==========================================
    # GET VALUES FROM HTML FORM
    # ==========================================

    gender = request.form["gender"]

    own_car = request.form["own_car"]

    own_house = request.form["own_house"]

    income = float(
        request.form["income"]
    )

    children = int(
        request.form["children"]
    )

    education = request.form["education"]

    occupation = request.form["occupation"]

    family_members = int(
        request.form["family_members"]
    )

    age = int(
        request.form["age"]
    )

    employment_years = int(
        request.form["employment_years"]
    )

    income_type = request.form["income_type"]

    family_status = request.form["family_status"]

    housing_type = request.form["housing_type"]


    # ==========================================
    # CONVERT AGE AND EMPLOYMENT
    # ==========================================

    days_birth = -(age * 365)

    days_employed = -(
        employment_years * 365
    )


    # ==========================================
    # CREATE INPUT DATAFRAME
    # ==========================================

    input_df = pd.DataFrame({

        "CODE_GENDER": [
            gender
        ],

        "FLAG_OWN_CAR": [
            own_car
        ],

        "FLAG_OWN_REALTY": [
            own_house
        ],

        "AMT_INCOME_TOTAL": [
            income
        ],

        "CNT_CHILDREN": [
            children
        ],

        "NAME_EDUCATION_TYPE": [
            education
        ],

        "OCCUPATION_TYPE": [
            occupation
        ],

        "CNT_FAM_MEMBERS": [
            family_members
        ],

        "NAME_INCOME_TYPE": [
            income_type
        ],

        "NAME_FAMILY_STATUS": [
            family_status
        ],

        "NAME_HOUSING_TYPE": [
            housing_type
        ],

        "DAYS_BIRTH": [
            days_birth
        ],

        "DAYS_EMPLOYED": [
            days_employed
        ]

    })


    # ==========================================
    # ENCODE INPUT
    # ==========================================

    input_df = pd.get_dummies(
        input_df
    )


    # ==========================================
    # ADD MISSING FEATURE COLUMNS
    # ==========================================

    for col in feature_columns:

        if col not in input_df.columns:

            input_df[col] = 0


    # ==========================================
    # KEEP SAME COLUMN ORDER
    # ==========================================

    input_df = input_df[
        feature_columns
    ]


    # ==========================================
    # DISPLAY MODEL INPUT
    # ==========================================

    print(
        "\n========== INPUT TO MODEL =========="
    )

    print(
        input_df
    )


    # ==========================================
    # MODEL PROBABILITY
    # ==========================================

    probability = model.predict_proba(
        input_df
    )[0]


    class_0_index = list(
        model.classes_
    ).index(0)

    class_1_index = list(
        model.classes_
    ).index(1)


    approval_probability = (
        probability[class_0_index] * 100
    )

    rejection_probability = (
        probability[class_1_index] * 100
    )


    print(
        "\n========== MODEL OUTPUT =========="
    )

    print(
        f"ML Approval Probability : "
        f"{approval_probability:.2f}%"
    )

    print(
        f"ML Rejection Probability: "
        f"{rejection_probability:.2f}%"
    )


    # ==========================================
    # COMPULSORY REJECTION RULES
    # ==========================================

    rejection_reasons = []


    # ==========================================
    # RULE 1
    # AGE BELOW 20 = REJECTED
    # ==========================================

    if age < 20:

        rejection_reasons.append(
            "Applicant age is below 20 years"
        )


    # ==========================================
    # RULE 2
    # RENTED HOUSE = REJECTED
    # ==========================================

    housing_value = str(
        housing_type
    ).strip().lower()


    if housing_value in [

        "rented apartment",

        "rented house",

        "rented",

        "rent"

    ]:

        rejection_reasons.append(
            "Applicant lives in rented housing"
        )


    # ==========================================
    # RULE 3
    # INCOME BELOW 20000 = REJECTED
    # ==========================================

    if income < 20000:

        rejection_reasons.append(
            "Annual income is below 20000"
        )


    # ==========================================
    # FINAL DECISION
    # ==========================================

    if len(rejection_reasons) > 0:

        # COMPULSORY REJECTION

        prediction = 1

        decision = (
            "❌ CREDIT CARD REJECTED"
        )

        risk = "HIGH RISK"


        # Demo policy decision score

        approval_probability = 20.0

        rejection_probability = 80.0


    else:

        # ======================================
        # REMAINING PROFILES:
        # RANDOM FOREST DECIDES AS IT IS
        # ======================================

        prediction = model.predict(
            input_df
        )[0]


        if prediction == 0:

            decision = (
                "✅ CREDIT CARD APPROVED"
            )

            risk = "LOW RISK"


        else:

            decision = (
                "❌ CREDIT CARD REJECTED"
            )

            risk = "HIGH RISK"


    # ==========================================
    # DEBUG OUTPUT
    # ==========================================

    print(
        "\n========== FINAL DECISION =========="
    )

    print(
        "Age:",
        age
    )

    print(
        "Annual Income:",
        income
    )

    print(
        "Housing Type:",
        housing_type
    )

    print(
        "Rejection Reasons:",
        rejection_reasons
    )

    print(
        "Prediction Class:",
        prediction
    )

    print(
        "Final Decision:",
        decision
    )

    print(
        f"Approval Probability: "
        f"{approval_probability:.2f}%"
    )

    print(
        f"Rejection Probability: "
        f"{rejection_probability:.2f}%"
    )

    print(
        "===================================="
    )


    # ==========================================
    # RESULT PAGE
    # ==========================================

    return render_template(

        "result.html",

        decision=decision,

        approval=round(
            approval_probability,
            2
        ),

        rejection=round(
            rejection_probability,
            2
        ),

        risk=risk,

        model_name="Random Forest",

        gender=gender,

        age=age,

        income=income,

        income_type=income_type,

        education=education,

        occupation=occupation,

        employment_experience=employment_years,

        own_car=own_car,

        own_house=own_house,

        housing_type=housing_type,

        family_status=family_status,

        children=children,

        family_members=family_members

    )


# ==========================================
# RUN APP
# ==========================================

if __name__ == "__main__":

    app.run(
        debug=True
    )