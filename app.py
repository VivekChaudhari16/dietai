"""
DietAI - AI-Powered Diet Recommendation System
Interactive Streamlit Dashboard

Run locally:
    streamlit run app.py

Deploy:
    Push this repo to GitHub, then deploy free at https://share.streamlit.io
    (Streamlit Community Cloud) pointing at app.py
"""

import json  # used to load the saved model comparison metrics (models/model_results.json)

import joblib          # loads/saves the trained sklearn model, scaler & encoders (.pkl files)
import numpy as np      # numeric operations (not heavily used directly, but common in ML apps)
import pandas as pd     # dataframe handling for the dataset and model input rows
import plotly.express as px       # quick, high-level interactive charts (bars, box, scatter, heatmap)
import plotly.graph_objects as go # imported for flexibility; px covers most charts used here
import streamlit as st  # the web app framework itself — every st.* call renders a UI element

# ----------------------------------------------------------------------------
# PAGE CONFIG
# Must be the first Streamlit command in the script — sets the browser tab
# title/icon and switches the app to a wide (full-width) layout.
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="DietAI | Diet Recommendation System",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# STYLING
# Custom CSS injected via st.markdown(unsafe_allow_html=True). Streamlit
# doesn't expose a native way to fully theme components, so raw CSS is the
# standard workaround for a polished look (gradient header, metric cards,
# the green "result" card shown after a prediction).
# ----------------------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #10b981, #059669);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header {
        color: #6b7280;
        font-size: 1.05rem;
        margin-top: 0;
    }
    div[data-testid="stMetric"] {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 12px;
        padding: 14px 16px;
    }
    .diet-card {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 28px;
        border-radius: 16px;
        text-align: center;
        margin-top: 10px;
    }
    .diet-card h2 { margin: 0; font-size: 1.6rem; }
    .diet-card p { margin: 6px 0 0 0; opacity: 0.9; }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# DATA / MODEL LOADING (cached)
# @st.cache_data / @st.cache_resource make sure the CSV and model files are
# only read from disk ONCE per session, not on every widget interaction —
# Streamlit reruns the whole script top-to-bottom on every user action, so
# caching here is what keeps the app fast.
#   - cache_data   -> for data that can be copied/serialized (DataFrames)
#   - cache_resource -> for objects that shouldn't be copied (ML models, connections)
# ----------------------------------------------------------------------------
@st.cache_data
def load_data():
    """Load the training dataset (used for the Data Explorer page & feature stats)."""
    return pd.read_csv("data/raw_data.csv")


@st.cache_resource
def load_model_artifacts():
    """Load the trained model + all preprocessing objects saved by train_model.py.

    Returns:
        model         -> the winning classifier (RandomForest/XGBoost/GradientBoosting)
        scaler        -> StandardScaler fitted on training features (must reuse, not refit)
        encoders      -> dict of LabelEncoders for each categorical column + the target
        feature_cols  -> exact column order the model expects (order matters!)
        results       -> dict of cross-validation/test metrics for all 3 compared models
    """
    model = joblib.load("models/best_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    encoders = joblib.load("models/encoders.pkl")
    feature_cols = joblib.load("models/feature_cols.pkl")
    with open("models/model_results.json") as f:
        results = json.load(f)
    return model, scaler, encoders, feature_cols, results


# Load once at app startup (cached above, so cheap on reruns)
df = load_data()
model, scaler, encoders, feature_cols, results = load_model_artifacts()

# Human-friendly labels/descriptions for each of the 7 diet classes the model
# predicts. Keyed by the exact string produced during data generation/labeling
# (see src/generate_data.py -> assign_diet()), so this must stay in sync with
# encoders["target"].classes_.
DIET_INFO = {
    "Diabetic-Friendly Low-Carb Diet": {
        "emoji": "🩸",
        "desc": "Low glycemic-index foods, controlled carbohydrate intake, high fiber. Focus on whole grains, legumes, and non-starchy vegetables.",
    },
    "Heart-Healthy Low-Cholesterol Diet": {
        "emoji": "❤️",
        "desc": "Reduced saturated fat, more omega-3s, lean proteins, and soluble fiber to support cardiovascular health.",
    },
    "Weight-Loss Calorie-Deficit Diet": {
        "emoji": "⚖️",
        "desc": "Structured calorie deficit with high protein and fiber to preserve muscle and manage hunger.",
    },
    "High-Calorie Weight-Gain Diet": {
        "emoji": "💪",
        "desc": "Nutrient-dense, calorie-surplus meals with balanced macros to support healthy weight gain.",
    },
    "Low-Impact Balanced Diet": {
        "emoji": "🌿",
        "desc": "Moderate calories with anti-inflammatory foods, suited for lower activity levels.",
    },
    "High-Protein Athletic Diet": {
        "emoji": "🏋️",
        "desc": "Higher protein and calorie needs to support high activity levels and muscle recovery.",
    },
    "Balanced Maintenance Diet": {
        "emoji": "🍽️",
        "desc": "Well-rounded macros across all food groups to maintain current health status.",
    },
}

# ----------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------
st.markdown('<p class="main-header">🥗 DietAI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Personalized Diet Recommendation System — built on Random Forest, XGBoost & Gradient Boosting</p>', unsafe_allow_html=True)
st.markdown("---")

# ----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# st.sidebar.* pins these widgets to the left nav panel instead of the main
# body. The radio button acts as a simple multi-page router — whichever
# option is selected controls which `if/elif` block below gets rendered.
# ----------------------------------------------------------------------------
page = st.sidebar.radio(
    "Navigate",
    ["🔮 Get Recommendation", "📊 Data Explorer", "🤖 Model Performance", "ℹ️ About"],
)

# Quick at-a-glance stats, always visible regardless of which page is open
st.sidebar.markdown("---")
st.sidebar.metric("Dataset Size", f"{len(df):,} records")
st.sidebar.metric("Best Model", results["best_model"])
st.sidebar.metric("Test Accuracy", f"{results['results'][results['best_model']]['test_accuracy']*100:.1f}%")

# ============================================================================
# PAGE 1: PREDICTION
# The core interactive feature — collects health inputs via sliders/dropdowns,
# recreates the SAME feature engineering used during training, then feeds the
# row through the saved scaler + model to get a live prediction.
# ============================================================================
if page == "🔮 Get Recommendation":
    st.subheader("Enter Your Health Profile")

    # 3-column layout just for visual grouping of related inputs
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("Age", 18, 80, 32)
        gender = st.selectbox("Gender", ["Male", "Female"])
        height = st.slider("Height (cm)", 140, 210, 170)
    with col2:
        weight = st.slider("Weight (kg)", 35, 160, 70)
        cholesterol = st.slider("Cholesterol (mg/dL)", 120, 320, 190)
        blood_sugar = st.slider("Fasting Blood Sugar (mg/dL)", 65, 250, 95)
    with col3:
        systolic_bp = st.slider("Systolic Blood Pressure", 90, 190, 120)
        activity_level = st.selectbox(
            "Activity Level", ["Sedentary", "Low", "Moderate", "High", "Very High"], index=2
        )

    # ---- Derived features (must exactly mirror src/train_model.py) ----
    bmi = weight / ((height / 100) ** 2)  # standard BMI formula: kg / m^2

    # Bin BMI/cholesterol/blood sugar into the same clinical risk categories
    # the model was trained on. pd.cut bins must use IDENTICAL edges/labels
    # to train_model.py, otherwise the encoder.transform() call below fails
    # or (worse) silently encodes the wrong category.
    bmi_cat = pd.cut([bmi], bins=[0, 18.5, 25, 30, 100],
                      labels=["Underweight", "Normal", "Overweight", "Obese"])[0]
    chol_risk = pd.cut([cholesterol], bins=[0, 200, 240, 500],
                        labels=["Normal", "Borderline", "High"])[0]
    sugar_risk = pd.cut([blood_sugar], bins=[0, 100, 126, 500],
                         labels=["Normal", "Prediabetic", "Diabetic"])[0]

    # Live preview metrics so the user sees their risk bands before predicting
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("BMI", f"{bmi:.1f}", bmi_cat)
    m2.metric("Cholesterol Risk", chol_risk)
    m3.metric("Blood Sugar Risk", sugar_risk)
    m4.metric("Activity", activity_level)

    st.markdown("")
    if st.button("🔮 Generate My Diet Recommendation", type="primary", use_container_width=True):
        # Composite "metabolic load" feature — same formula as train_model.py.
        # Normalizes each raw value against the training-set mean so the three
        # metrics (different units/scales) combine into one comparable score.
        metabolic_load = (
            (bmi / df["BMI"].mean())
            + (cholesterol / df["Cholesterol"].mean())
            + (blood_sugar / df["BloodSugar"].mean())
        )

        # Build a single-row feature dict. Categorical values are converted to
        # integers via the SAME LabelEncoders fitted during training
        # (encoders["Gender"].transform, etc.) — reusing .transform() (not
        # .fit_transform()) is critical so category->number mapping matches.
        row = {
            "Age": age,
            "Gender_enc": encoders["Gender"].transform([gender])[0],
            "Height_cm": height,
            "Weight_kg": weight,
            "BMI": bmi,
            "Cholesterol": cholesterol,
            "BloodSugar": blood_sugar,
            "SystolicBP": systolic_bp,
            "ActivityLevel_enc": encoders["ActivityLevel"].transform([activity_level])[0],
            "BMI_Category_enc": encoders["BMI_Category"].transform([str(bmi_cat)])[0],
            "Cholesterol_Risk_enc": encoders["Cholesterol_Risk"].transform([str(chol_risk)])[0],
            "Sugar_Risk_enc": encoders["Sugar_Risk"].transform([str(sugar_risk)])[0],
            "Metabolic_Load": metabolic_load,
        }
        # [feature_cols] enforces the exact column ORDER the model was trained
        # on — dicts don't guarantee order across Python versions, so this is
        # a safety net against silently feeding columns in the wrong order.
        X_input = pd.DataFrame([row])[feature_cols]
        X_scaled = scaler.transform(X_input)  # reuse fitted scaler, don't refit

        pred_idx = model.predict(X_scaled)[0]                              # predicted class index (int)
        pred_label = encoders["target"].inverse_transform([pred_idx])[0]   # -> human-readable diet name
        proba = model.predict_proba(X_scaled)[0]                           # confidence per class

        info = DIET_INFO.get(pred_label, {"emoji": "🍎", "desc": ""})
        st.markdown(f"""
        <div class="diet-card">
            <h2>{info['emoji']} {pred_label}</h2>
            <p>{info['desc']}</p>
        </div>
        """, unsafe_allow_html=True)

        # Horizontal bar chart of predict_proba() across all 7 classes —
        # shows not just the winning label but how confident/close the call was.
        st.markdown("#### Prediction Confidence")
        proba_df = pd.DataFrame({
            "Diet Plan": encoders["target"].classes_,
            "Confidence": proba,
        }).sort_values("Confidence", ascending=True)
        fig = px.bar(
            proba_df, x="Confidence", y="Diet Plan", orientation="h",
            color="Confidence", color_continuous_scale="Greens",
        )
        fig.update_layout(height=350, showlegend=False, xaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 2: DATA EXPLORER
# Read-only exploratory data analysis (EDA) view over the raw training
# dataset — lets a viewer sanity-check the data the model was trained on.
# ============================================================================
elif page == "📊 Data Explorer":
    st.subheader("Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("Features", f"{df.shape[1]-1}")
    c3.metric("Diet Classes", df["DietRecommendation"].nunique())
    c4.metric("Avg BMI", f"{df['BMI'].mean():.1f}")

    st.markdown("#### Sample Data")
    st.dataframe(df.head(20), use_container_width=True)

    st.markdown("#### Diet Recommendation Distribution")
    dist = df["DietRecommendation"].value_counts().reset_index()
    dist.columns = ["Diet", "Count"]
    fig1 = px.bar(dist, x="Count", y="Diet", orientation="h", color="Count",
                  color_continuous_scale="Greens")
    fig1.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)

    colA, colB = st.columns(2)
    with colA:
        # Box plot: shows how BMI spreads/differs across the 7 diet classes —
        # a quick visual check that the labels correlate sensibly with BMI.
        st.markdown("#### BMI Distribution by Diet")
        fig2 = px.box(df, x="DietRecommendation", y="BMI", color="DietRecommendation")
        fig2.update_layout(height=420, showlegend=False, xaxis_tickangle=-35)
        st.plotly_chart(fig2, use_container_width=True)
    with colB:
        # Scatter plot on a random sample (not the full 10k rows) to keep the
        # chart responsive — 1500 points is plenty to see the clustering.
        st.markdown("#### Cholesterol vs Blood Sugar")
        fig3 = px.scatter(
            df.sample(min(1500, len(df))), x="Cholesterol", y="BloodSugar",
            color="DietRecommendation", opacity=0.6,
        )
        fig3.update_layout(height=420)
        st.plotly_chart(fig3, use_container_width=True)

    # Correlation heatmap over the raw numeric columns — helps explain WHY
    # certain features (e.g. BMI vs Weight) end up highly predictive/redundant.
    st.markdown("#### Feature Correlation Heatmap")
    numeric_cols = ["Age", "Height_cm", "Weight_kg", "BMI", "Cholesterol", "BloodSugar", "SystolicBP"]
    corr = df[numeric_cols].corr()
    fig4 = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdYlGn", aspect="auto")
    fig4.update_layout(height=450)
    st.plotly_chart(fig4, use_container_width=True)

# ============================================================================
# PAGE 3: MODEL PERFORMANCE
# Surfaces the metrics saved by src/train_model.py (models/model_results.json)
# so the model-selection process is transparent rather than a black box.
# ============================================================================
elif page == "🤖 Model Performance":
    st.subheader("Model Comparison: Random Forest vs XGBoost vs Gradient Boosting")

    # results["results"] is a dict-of-dicts: {model_name: {metric: value}}.
    # Transposing it (.T) turns model names into rows, metrics into columns.
    res = results["results"]
    comp_df = pd.DataFrame(res).T.reset_index().rename(columns={"index": "Model"})
    comp_df["test_accuracy"] = comp_df["test_accuracy"].astype(float)
    comp_df["test_f1_weighted"] = comp_df["test_f1_weighted"].astype(float)
    comp_df["cv_f1_weighted"] = comp_df["cv_f1_weighted"].astype(float)

    st.success(f"🏆 Best performing model: **{results['best_model']}**")

    c1, c2, c3 = st.columns(3)
    for i, row in comp_df.iterrows():
        col = [c1, c2, c3][i % 3]
        col.metric(
            row["Model"],
            f"{row['test_accuracy']*100:.2f}% acc",
            f"F1: {row['test_f1_weighted']:.3f}",
        )

    # Grouped bar chart comparing cross-validation F1 (5-fold, seen only
    # during training) against held-out test F1 (unseen 20% split) — a big
    # gap between the two would signal overfitting.
    st.markdown("#### Cross-Validated F1 vs Test F1")
    melt = comp_df.melt(id_vars="Model", value_vars=["cv_f1_weighted", "test_f1_weighted"],
                         var_name="Metric", value_name="Score")
    fig = px.bar(melt, x="Model", y="Score", color="Metric", barmode="group",
                 color_discrete_sequence=["#10b981", "#059669"])
    fig.update_layout(height=400, yaxis_range=[0.9, 1.0])
    st.plotly_chart(fig, use_container_width=True)

    # Show the winning hyperparameter combination GridSearchCV found for
    # each model (e.g. n_estimators, max_depth, learning_rate).
    st.markdown("#### Best Hyperparameters Found (GridSearchCV)")
    for _, row in comp_df.iterrows():
        with st.expander(f"{row['Model']} — best params"):
            st.json(row["best_params"])

    # Tree-based models (RF/XGBoost/GradientBoosting) all expose
    # feature_importances_ after fitting — ranks which inputs most influenced
    # the model's decisions (e.g. BMI or BloodSugar likely dominate here).
    st.markdown("#### Feature Importance (Best Model)")
    if hasattr(model, "feature_importances_"):
        imp_df = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": model.feature_importances_,
        }).sort_values("Importance", ascending=True)
        fig2 = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                      color="Importance", color_continuous_scale="Greens")
        fig2.update_layout(height=450, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

# ============================================================================
# PAGE 4: ABOUT
# ============================================================================
else:
    st.subheader("About DietAI")
    st.markdown("""
    **DietAI** is an end-to-end machine learning system that recommends a personalized
    diet plan based on a person's health profile — BMI, cholesterol, blood sugar,
    blood pressure, and activity level.

    **Pipeline:**
    1. **Data** — 10,000-record health dataset (BMI, cholesterol, blood sugar, activity level, etc.)
    2. **Feature Engineering** — clinical BMI/cholesterol/blood-sugar risk bands + a composite metabolic load score
    3. **Model Training** — Random Forest, XGBoost, and Gradient Boosting compared via 5-fold
       cross-validation and `GridSearchCV` hyperparameter tuning
    4. **Deployment** — best model served through this interactive Streamlit dashboard

    **Tech stack:** Python, Scikit-learn, XGBoost, Pandas, NumPy, Streamlit, Plotly

    ---
    ⚠️ **Disclaimer:** This tool is for educational/portfolio purposes only. It is not
    a substitute for medical advice — always consult a registered dietitian or physician
    before making dietary changes.
    """)
