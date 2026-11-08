"""
Streamlit app: Student Dropout & Academic Success — classifier evaluation.

The app NEVER retrains. It only:
  1. loads the 5 pipelines saved by model/train_models.ipynb,
  2. reads an uploaded test CSV (features + 'Target' column),
  3. predicts with the saved models and shows metrics, confusion matrix, report.

Run locally:  streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from utils import METRIC_NAMES, clean_columns, evaluate

# ---------------------------------------------------------------------------
# Paths are relative to THIS file, so the app works no matter which folder
# it is launched from (important on Streamlit Community Cloud).
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"
SAMPLE_TEST_FILE = BASE_DIR / "test_data.csv"

st.set_page_config(
    page_title="Student Outcome Classifier",
    page_icon="🎓",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Loading helpers (cached so files are read only once per server session)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_metadata():
    with open(MODEL_DIR / "metadata.json") as f:
        return json.load(f)


@st.cache_resource
def load_models(model_files):
    """Load every saved pipeline: {display name: fitted Pipeline}."""
    # model_files is passed as a tuple of pairs so Streamlit can hash it
    return {name: joblib.load(MODEL_DIR / fname) for name, fname in model_files}


def read_csv(file_or_path):
    """Read a CSV; sep=None lets pandas detect ',' or ';' automatically,
    so both our test_data.csv and the original UCI file format work."""
    df = pd.read_csv(file_or_path, sep=None, engine="python", encoding="utf-8-sig")
    return clean_columns(df)


def fmt(value):
    return "n/a" if pd.isna(value) else f"{value:.4f}"


# ---------------------------------------------------------------------------
# Load artifacts
# ---------------------------------------------------------------------------
try:
    meta = load_metadata()
    models = load_models(tuple(meta["model_files"].items()))
except FileNotFoundError:
    st.error(
        "Saved models not found in the `model/` folder. "
        "Run `model/train_models.ipynb` first to create the .joblib files."
    )
    st.stop()

FEATURES = meta["feature_columns"]
TARGET = meta["target"]
CLASS_NAMES = meta["class_names"]
LABEL_MAP = meta["label_mapping"]          # e.g. {"Dropout": 0, "Enrolled": 1, ...}

# ---------------------------------------------------------------------------
# Sidebar: dataset info + model selection
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("🎓 About the data")
    st.markdown(
        f"""
**UCI – Predict Students' Dropout and Academic Success**

Records of students enrolled in undergraduate degrees at a Portuguese
higher-education institution: demographics, socio-economic factors,
admission data and 1st/2nd semester academic performance.

- **Features:** {len(FEATURES)}
- **Target:** `{TARGET}` → {", ".join(CLASS_NAMES)}
- **Train / test rows:** {meta["n_train"]} / {meta["n_test"]}
"""
    )
    st.divider()

    st.header("⚙️ Model")
    selected_model = st.selectbox("Choose a classifier", list(models.keys()))

    st.divider()
    # Let evaluators grab the exact held-out test file used in the README
    if SAMPLE_TEST_FILE.exists():
        st.download_button(
            "⬇️ Download sample test_data.csv",
            data=SAMPLE_TEST_FILE.read_bytes(),
            file_name="test_data.csv",
            mime="text/csv",
        )

# ---------------------------------------------------------------------------
# Header + upload
# ---------------------------------------------------------------------------
st.title("Will the student drop out, stay enrolled, or graduate?")
st.caption(
    "Upload held-out test data and see how five pre-trained classifiers perform. "
    "Models were trained once in the notebook; this app only loads them and predicts."
)

col_up, col_opt = st.columns([3, 1])
with col_up:
    uploaded = st.file_uploader(
        "Upload test CSV (all feature columns + the `Target` column)", type=["csv"]
    )
with col_opt:
    st.write("")
    st.write("")
    use_sample = st.toggle("Use bundled test_data.csv", value=uploaded is None)

if uploaded is not None:
    data = read_csv(uploaded)
    source = uploaded.name
elif use_sample and SAMPLE_TEST_FILE.exists():
    data = read_csv(SAMPLE_TEST_FILE)
    source = "test_data.csv (bundled)"
else:
    st.info("⬆️ Upload a CSV file, or switch on *Use bundled test_data.csv*.")
    st.stop()

# ---------------------------------------------------------------------------
# Validate the uploaded columns and labels
# ---------------------------------------------------------------------------
missing = [c for c in FEATURES + [TARGET] if c not in data.columns]
if missing:
    st.error(
        f"The file is missing {len(missing)} required column(s):\n\n"
        + "\n".join(f"- `{c}`" for c in missing)
    )
    st.stop()

data = data.dropna(subset=[TARGET])
labels = data[TARGET].astype(str).str.strip()
unknown = sorted(set(labels) - set(LABEL_MAP))
if unknown:
    st.error(
        f"Unknown target value(s) {unknown}. Expected one of: {', '.join(CLASS_NAMES)}."
    )
    st.stop()

X = data[FEATURES]                       # same column order as during training
y = labels.map(LABEL_MAP).to_numpy()     # text label -> integer 0..n-1

st.success(f"Loaded **{source}**: {len(data)} rows, {data.shape[1]} columns.")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_single, tab_compare, tab_data = st.tabs(
    ["📊 Selected model", "🏆 Compare all models", "🔎 Data preview"]
)

# ---- Tab 1: one model in detail -------------------------------------------
with tab_single:
    model = models[selected_model]
    metrics, y_pred, y_proba = evaluate(model, X, y)

    st.subheader(f"{selected_model} — evaluation metrics")
    for col, name in zip(st.columns(len(METRIC_NAMES)), METRIC_NAMES):
        col.metric(name, fmt(metrics[name]))
    st.caption(
        "Precision, Recall and F1 are weighted averages over the classes; "
        "AUC is one-vs-rest. MCC ranges from -1 to 1 (0 = random guessing)."
    )

    left, right = st.columns(2)

    with left:
        st.markdown("**Confusion matrix**")
        cm = confusion_matrix(y, y_pred, labels=list(range(len(CLASS_NAMES))))
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Diagonal = correct predictions. Off-diagonal = confusions.")

    with right:
        st.markdown("**Classification report**")
        report = classification_report(
            y, y_pred,
            labels=list(range(len(CLASS_NAMES))),
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )
        report_df = pd.DataFrame(report).T
        report_df["support"] = report_df["support"].astype(int)
        st.dataframe(report_df.round(4))

    # Predictions download: original rows + actual, predicted, class probabilities
    pred_df = data.copy()
    pred_df["Predicted"] = [CLASS_NAMES[i] for i in y_pred]
    pred_df["Correct"] = pred_df["Predicted"] == labels.values
    for i, cls in enumerate(CLASS_NAMES):
        pred_df[f"P({cls})"] = y_proba[:, i].round(4)

    st.download_button(
        f"⬇️ Download {selected_model} predictions (CSV)",
        data=pred_df.to_csv(index=False).encode("utf-8"),
        file_name=f"predictions_{selected_model.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )

# ---- Tab 2: all models side by side ----------------------------------------
with tab_compare:
    st.subheader("All five models on the same data")
    rows = []
    for name, mdl in models.items():
        m, _, _ = evaluate(mdl, X, y)
        rows.append({"Model": name, **m})
    comp = pd.DataFrame(rows).set_index("Model")[METRIC_NAMES]

    # Highlight the best score in every metric column
    st.dataframe(
        comp.style.format("{:.4f}").highlight_max(axis=0, color="#c8e6c9"),
    )

    best = comp["F1"].idxmax()
    st.markdown(
        f"🏆 Best F1 on this file: **{best}** (F1 = {comp.loc[best, 'F1']:.4f}, "
        f"MCC = {comp.loc[best, 'MCC']:.4f})"
    )

    # Grouped bar chart: one group per metric, one bar per model
    fig, ax = plt.subplots(figsize=(10, 4))
    comp.T.plot(kind="bar", ax=ax, width=0.8, colormap="tab10")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_xticklabels(METRIC_NAMES, rotation=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=5, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.download_button(
        "⬇️ Download comparison table (CSV)",
        data=comp.round(4).to_csv().encode("utf-8"),
        file_name="model_comparison.csv",
        mime="text/csv",
    )

# ---- Tab 3: look at the uploaded data --------------------------------------
with tab_data:
    st.subheader("Uploaded data")
    st.dataframe(data.head(50))
    st.markdown("**Class distribution in this file**")
    st.bar_chart(labels.value_counts().reindex(CLASS_NAMES, fill_value=0))
