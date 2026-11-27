
import streamlit as st
import pandas as pd
import joblib
import json
from pathlib import Path

from src.data.preprocessor import clean_text


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>
.stApp {
    background: #f3f6fc;
    color: #172b4d;
}

.main .block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(18px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes gradientMove {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

@keyframes resultAppear {
    from { opacity: 0; transform: scale(0.94); }
    to { opacity: 1; transform: scale(1); }
}

.hero {
    background: linear-gradient(120deg, #12366c, #285bd0, #6845bd);
    background-size: 200% 200%;
    animation: gradientMove 8s ease infinite, fadeUp 0.8s ease;
    padding: 38px;
    border-radius: 22px;
    color: white;
    box-shadow: 0 12px 30px #315fd433;
}

.hero h1 {
    color: white !important;
    font-size: 40px;
    font-weight: 900;
    letter-spacing: 1px;
}

.hero p {
    color: #e5edff !important;
    font-size: 16px;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #10284f, #172e5c) !important;
}

section[data-testid="stSidebar"] * {
    color: #edf4ff !important;
}

section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    color: #c0d2ef !important;
}

section[data-testid="stSidebar"] .status-ready,
section[data-testid="stSidebar"] .status-ready * {
    background: #d1fae5 !important;
    color: #065f46 !important;
    -webkit-text-fill-color: #065f46 !important;
    font-weight: 900 !important;
}

section[data-testid="stSidebar"] .status-error,
section[data-testid="stSidebar"] .status-error * {
    background: #fee2e2 !important;
    color: #991b1b !important;
    -webkit-text-fill-color: #991b1b !important;
    font-weight: 900 !important;
}

.status-ready, .status-error {
    padding: 17px;
    border-radius: 12px;
    font-size: 16px;
    border: 1px solid #6ee7b7;
}

h1, h2, h3 {
    color: #18345d;
}

.stTextArea label {
    color: #18345d !important;
    font-weight: 700 !important;
    font-size: 15px !important;
}

.stTextArea textarea {
    background: white !important;
    color: #17233c !important;
    -webkit-text-fill-color: #17233c !important;
    caret-color: #245bd5 !important;
    border: 1.5px solid #bdcce4 !important;
    border-radius: 14px !important;
    font-size: 16px !important;
    padding: 16px !important;
}

.stTextArea textarea::placeholder {
    color: #75849d !important;
    -webkit-text-fill-color: #75849d !important;
    opacity: 1 !important;
}

div.stButton > button {
    background: linear-gradient(100deg, #245bd5, #6847ce);
    color: white !important;
    border: none;
    border-radius: 11px;
    padding: 12px 22px;
    font-weight: 700;
    transition: all 0.25s ease;
}

div.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 20px #315fd455;
}

div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e1e8f3;
    padding: 18px;
    border-radius: 14px;
    box-shadow: 0 4px 15px #19365b0b;
    transition: 0.25s ease;
}

div[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
}

.result-card {
    background: white;
    padding: 30px;
    border-radius: 18px;
    text-align: center;
    border: 1px solid #e0e8f5;
    box-shadow: 0 8px 25px #19365b12;
    animation: resultAppear 0.6s ease;
}

.real-result {
    color: #07966b;
    font-size: 32px;
    font-weight: 900;
}

.fake-result {
    color: #dc4054;
    font-size: 32px;
    font-weight: 900;
}

.info-card {
    background: white;
    border: 1px solid #e1e8f3;
    border-radius: 16px;
    padding: 23px;
    box-shadow: 0 5px 18px #19365b0b;
    animation: fadeUp 0.7s ease;
}

.info-card h3 {
    color: #18345d !important;
}

.info-card p {
    color: #53647e !important;
}

.footer {
    text-align: center;
    color: #7888a2;
    padding: 22px;
    font-size: 13px;
}

hr {
    border-color: #dce5f1;
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# LOAD MIXED-INPUT TRAINED MODEL
# =====================================================

@st.cache_resource
def load_model():

    model_path = Path("models/linearsvm_mixed.joblib")
    vectorizer_path = Path("models/tfidf_vectorizer_mixed.joblib")

    if not model_path.exists():
        raise FileNotFoundError("Mixed-input model not found.")

    if not vectorizer_path.exists():
        raise FileNotFoundError("Mixed-input vectorizer not found.")

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    return model, vectorizer


try:
    model, vectorizer = load_model()
    model_loaded = True
    model_error = None

except Exception as error:
    model_loaded = False
    model_error = str(error)


# =====================================================
# MAIN HEADER
# =====================================================

st.markdown("""
<div class="hero">
<h1>🔎 FAKE NEWS DETECTION SYSTEM</h1>
<p>AI-Powered News Credibility Analysis</p>
<p>Analyze news headlines, claims and articles using Natural Language Processing and Machine Learning.</p>
</div>
""", unsafe_allow_html=True)

st.write("")


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.markdown(
    "<h2 style='color:white !important;'>🔎 FAKE NEWS DETECTION</h2>",
    unsafe_allow_html=True
)

st.sidebar.caption("NEWS ANALYSIS PLATFORM")
st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "News Analyzer",
        "Model Performance",
        "Project Information"
    ]
)

st.sidebar.divider()
st.sidebar.markdown("### SYSTEM STATUS")

if model_loaded:
    st.sidebar.markdown(
        '<div class="status-ready">● MODEL IS READY</div>',
        unsafe_allow_html=True
    )
else:
    st.sidebar.markdown(
        '<div class="status-error">● MODEL NOT AVAILABLE</div>',
        unsafe_allow_html=True
    )
    st.sidebar.caption(model_error or "Unknown loading error")

st.sidebar.write("")
st.sidebar.caption("Dataset: ISOT")
st.sidebar.caption("Classifier: Mixed-input Linear SVM")
st.sidebar.caption("Features: Word + Character TF-IDF")


# =====================================================
# PAGE 1 - NEWS ANALYZER
# =====================================================

if page == "News Analyzer":

    st.write("")
    st.subheader("📰 News Credibility Analyzer")

    st.write(
        "Enter a headline, claim, sentence or complete news article."
    )

    news = st.text_area(
        "NEWS CONTENT",
        height=230,
        placeholder="Paste or type your news content here..."
    )

    analyze = st.button(
        "🔍 ANALYZE NEWS",
        type="primary",
        disabled=not model_loaded
    )

    if analyze:

        cleaned_news = clean_text(news)

        if not cleaned_news:
            st.warning("Please enter news content.")

        elif len(cleaned_news.split()) < 3:
            st.warning("Please enter at least 3 meaningful words.")

        elif (
            sum(c.isalpha() for c in cleaned_news)
            / max(len(cleaned_news), 1)
        ) < 0.55:
            st.warning(
                "Please enter readable text instead of random symbols."
            )

        else:

            with st.spinner("Analyzing news content..."):

                features = vectorizer.transform([cleaned_news])
                prediction = int(model.predict(features)[0])
                score = float(model.decision_function(features)[0])

            st.divider()
            st.subheader("ANALYSIS RESULT")

            if prediction == 1:

                st.markdown("""
<div class="result-card">
<div class="fake-result">⚠ LIKELY FAKE</div>
<p style="color:#53647e;">The model detected patterns associated with fake news in its training dataset.</p>
</div>
""", unsafe_allow_html=True)

            else:

                st.markdown("""
<div class="result-card">
<div class="real-result">✓ LIKELY REAL</div>
<p style="color:#53647e;">The model detected patterns associated with real news in its training dataset.</p>
</div>
""", unsafe_allow_html=True)

            st.write("")
            st.subheader("📊 TEXT STATISTICS")

            col1, col2, col3 = st.columns(3)

            col1.metric("Word Count", len(cleaned_news.split()))
            col2.metric("Character Count", len(cleaned_news))
            col3.metric(
                "Predicted Class",
                "FAKE" if prediction == 1 else "REAL"
            )

            st.write("")
            st.subheader("🧠 MODEL DECISION ANALYSIS")

            st.metric(
                "Linear SVM Decision Score",
                f"{score:.4f}"
            )

            st.info(
                "The SVM decision score is not a probability or "
                "confidence percentage. This prediction is based on "
                "patterns learned from the ISOT dataset. It does not "
                "independently verify the news."
            )


# =====================================================
# PAGE 2 - MODEL PERFORMANCE
# =====================================================

elif page == "Model Performance":

    st.subheader("📊 MODEL PERFORMANCE DASHBOARD")

    validation_file = Path(
        "reports/metrics/ml_model_comparison.csv"
    )

    test_file = Path(
        "reports/metrics/final_test_results.json"
    )

    mixed_validation_file = Path(
        "reports/metrics/mixed_input_validation.csv"
    )

    mixed_test_file = Path(
        "reports/metrics/mixed_input_test_results.json"
    )

    # Original validation comparison

    if validation_file.exists():

        validation = pd.read_csv(validation_file)

        st.markdown("### Original ML Model Validation")

        st.dataframe(
            validation,
            use_container_width=True,
            hide_index=True
        )

        if "model" in validation.columns:

            chart_data = validation.set_index("model")[
                ["accuracy", "precision", "recall", "f1_score"]
            ]

            st.markdown("### Original Model Comparison")
            st.bar_chart(chart_data)

    else:
        st.warning("Original validation metrics report not found.")

    st.divider()

    # Original model test results

    if test_file.exists():

        with open(test_file, "r", encoding="utf-8") as file:
            results = json.load(file)

        if isinstance(results, list):

            test_df = pd.DataFrame([
                {
                    "Model": item["model"],
                    "Accuracy": item["accuracy"],
                    "Precision": item["precision"],
                    "Recall": item["recall"],
                    "F1 Score": item["f1_score"],
                    "ROC-AUC": item.get("roc_auc")
                }
                for item in results
            ])

            st.markdown("### Original Models - Final Test Results")

            st.dataframe(
                test_df,
                use_container_width=True,
                hide_index=True
            )

            best = max(results, key=lambda item: item["f1_score"])

            st.success(
                f"Best Original Test Model: {best['model']} | "
                f"F1 Score: {best['f1_score']:.4f}"
            )

            st.markdown("### Original Model Confusion Matrix")

            matrix_df = pd.DataFrame(
                best["confusion_matrix"],
                index=["Actual REAL", "Actual FAKE"],
                columns=["Predicted REAL", "Predicted FAKE"]
            )

            st.dataframe(matrix_df, use_container_width=True)

            st.caption("Label mapping: 0 = REAL | 1 = FAKE")

    else:
        st.warning("Original final test results not found.")

    st.divider()

    # Mixed-input validation results

    st.markdown("### Mixed-Input Model Validation")

    if mixed_validation_file.exists():

        mixed_validation = pd.read_csv(mixed_validation_file)

        st.dataframe(
            mixed_validation,
            use_container_width=True,
            hide_index=True
        )

    else:
        st.warning("Mixed-input validation report not found.")

    st.divider()

    # Mixed-input final test results

    st.markdown("### Mixed-Input Model - Final Test Results")

    if mixed_test_file.exists():

        with open(mixed_test_file, "r", encoding="utf-8") as file:
            mixed_results = json.load(file)

        mixed_test_df = pd.DataFrame([
            {
                "Input Type": name.replace("_", " ").title(),
                "Accuracy": metrics["accuracy"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1 Score": metrics["f1_score"],
                "ROC-AUC": metrics["roc_auc"]
            }
            for name, metrics in mixed_results.items()
        ])

        st.dataframe(
            mixed_test_df,
            use_container_width=True,
            hide_index=True
        )

        for name, metrics in mixed_results.items():

            with st.expander(
                name.replace("_", " ").title() + " - Confusion Matrix"
            ):

                matrix_df = pd.DataFrame(
                    metrics["confusion_matrix"],
                    index=["Actual REAL", "Actual FAKE"],
                    columns=["Predicted REAL", "Predicted FAKE"]
                )

                st.dataframe(
                    matrix_df,
                    use_container_width=True
                )

        st.caption(
            "Mixed-input model was evaluated separately on "
            "headline-only and full-article inputs."
        )

    else:
        st.warning("Mixed-input test results not found.")


# =====================================================
# PAGE 3 - PROJECT INFORMATION
# =====================================================

elif page == "Project Information":

    st.subheader("ℹ️ ABOUT THE PROJECT")

    st.markdown("""
<div class="info-card">
<h3>Fake News Detection System</h3>
<p>This academic project uses Natural Language Processing and Machine Learning to identify patterns associated with fake and real news content.</p>
<p>The system supports both headline-only and full-article inputs through mixed-input training.</p>
</div>
""", unsafe_allow_html=True)

    st.write("")
    st.subheader("TECHNOLOGY STACK")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
<div class="info-card">
<h3>Natural Language Processing</h3>
<p>• Text Cleaning<br>• Text Normalization<br>• Word TF-IDF<br>• Character TF-IDF</p>
</div>
""", unsafe_allow_html=True)

    with col2:

        st.markdown("""
<div class="info-card">
<h3>Machine Learning</h3>
<p>• Logistic Regression<br>• Multinomial Naive Bayes<br>• Linear SVM<br>• Random Forest</p>
</div>
""", unsafe_allow_html=True)

    st.write("")
    st.subheader("PROJECT OBJECTIVES")

    st.markdown("""
- Analyze news content using NLP.
- Compare multiple ML algorithms.
- Train a model using both headlines and full articles.
- Evaluate accuracy, precision, recall and F1 score.
- Generate confusion matrices and ROC-AUC metrics.
- Provide an interactive news analysis interface.
""")

    st.subheader("MODEL LIMITATIONS")

    st.warning(
        "The model is trained and tested on the ISOT dataset. "
        "Its predictions identify learned text patterns and do not "
        "guarantee factual correctness. It does not perform "
        "live internet fact-checking."
    )


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.markdown("""
<div class="footer">
<b>FAKE NEWS DETECTION SYSTEM</b><br>
Academic Machine Learning Project<br>
NLP • TF-IDF • Mixed-Input Linear SVM<br><br>
Predictions require independent verification.
</div>
""", unsafe_allow_html=True)
