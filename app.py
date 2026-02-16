import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt


# ---------------- LOAD MODEL ----------------
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# create SHAP explainer
explainer = shap.TreeExplainer(model)


# ---------------- PREPROCESS FUNCTION ----------------
def preprocess_input(input_dict):

    df = pd.DataFrame([input_dict])

    # match training feature order
    model_features = model.feature_names_in_
    df = df[model_features]

    # apply encoders safely (no crash if unseen value)
    for col, encoder in encoders.items():
        if col in df.columns:
            try:
                df[col] = encoder.transform(df[col])
            except ValueError:
                df[col] = 0  # fallback value

    return df


# ---------------- PREDICTION FUNCTION ----------------
def predict_autism(input_data):

    processed = preprocess_input(input_data)

    probability = model.predict_proba(processed)[0][1]
    prediction = 1 if probability > 0.35 else 0

    if prediction == 1:
        return "Autism Detected", probability, processed
    else:
        return "No Autism Detected", probability, processed


# ---------------- UI START ----------------
st.title("Autism Prediction System")
st.write("Enter patient details")


# numeric input
age = st.number_input("Age", min_value=1, max_value=100, value=12)


# ---------- SAFE DROPDOWNS (MATCH TRAINING DATA) ----------
gender = st.selectbox("Gender", encoders["gender"].classes_)
ethnicity = st.selectbox("Ethnicity", encoders["ethnicity"].classes_)
jaundice = st.selectbox("Jaundice", encoders["jaundice"].classes_)
austim = st.selectbox("Family Autism History", encoders["austim"].classes_)
used_app = st.selectbox("Used App Before", encoders["used_app_before"].classes_)
relation = st.selectbox("Relation", encoders["relation"].classes_)

country = st.text_input("Country of Residence", "India")
result_score = st.slider("Screening Result Score", 0, 10, 5)


# A1–A10 scores
st.subheader("Screening Questions (A1–A10)")
scores = {}
for i in range(1, 11):
    scores[f"A{i}_Score"] = st.selectbox(f"A{i}_Score", [0, 1], key=f"A{i}")


# ---------------- CREATE INPUT DATA ----------------
input_data = {
    "age": age,
    "gender": gender,
    "ethnicity": ethnicity,
    "jaundice": jaundice,
    "austim": austim,
    "contry_of_res": country,
    "used_app_before": used_app,
    "result": result_score,
    "relation": relation,
    **scores
}


# ---------------- PREDICTION BUTTON ----------------
if st.button("Predict"):

    prediction_text, probability, processed = predict_autism(input_data)

    # show result
    if prediction_text == "Autism Detected":
        st.error(prediction_text)
    else:
        st.success(prediction_text)

    st.write("Prediction Probability:", round(probability * 100, 2), "%")

    # ---------------- SHAP EXPLANATION ----------------
    st.subheader("Why this prediction happened")

    shap_values = explainer(processed)

    fig = plt.figure()
    shap.plots.waterfall(shap_values[0, :, 1], show=False)
    st.pyplot(fig)
