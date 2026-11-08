# Student Dropout & Academic Success — ML Classification

**Live Streamlit app:** <PASTE_STREAMLIT_APP_LINK_HERE>

---

## a. Problem statement

Higher-education institutions lose students every year to dropout, and late
detection leaves little room to intervene. The goal of this project is to
predict, from information available at enrolment and after the first two
semesters, whether a student will **drop out**, **remain enrolled** beyond the
normal duration, or **graduate**. It is a **3-class classification** problem.

Five classifiers (Logistic Regression, Decision Tree, K-Nearest Neighbours,
Gaussian Naive Bayes and Support Vector Machine) are trained on the same data,
compared on six metrics (Accuracy, AUC, Precision, Recall, F1, MCC), and served
through an interactive Streamlit web app.

## b. Dataset description

| Item | Value |
|---|---|
| Name | Predict Students' Dropout and Academic Success |
| Source | UCI Machine Learning Repository — https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success |
| Instances | 4,424 students |
| Features | 36 (all numeric; categorical attributes are already integer-coded) |
| Missing values | None |
| Target column | `Target` |
| Classes | `Dropout` (1,421 · 32.1%), `Enrolled` (794 · 17.9%), `Graduate` (2,209 · 49.9%) |

The features fall into five groups:

- **Demographic:** marital status, nationality, gender, age at enrolment, displaced, international.
- **Socio-economic:** parents' qualifications and occupations, scholarship holder, debtor, tuition fees up to date, special educational needs.
- **Academic path at enrolment:** application mode and order, course, day/evening attendance, previous qualification and grade, admission grade.
- **Academic performance:** curricular units credited, enrolled, evaluated, approved, grade and without-evaluation for the 1st and 2nd semesters.
- **Macro-economic:** unemployment rate, inflation rate, GDP.

The classes are imbalanced (`Enrolled` is the minority), so the data is split
with a **stratified 80/20 split** (`random_state=42`), and Precision, Recall and
F1 are **weighted averages** over the classes. AUC is computed **one-vs-rest**.
The 885-row held-out test split is provided as `test_data.csv`.

## c. GitHub repository link

<PASTE_GITHUB_REPO_LINK_HERE>

```
├── app.py                     # Streamlit app (loads saved models, no retraining)
├── utils.py                   # evaluate(): shared metric code for notebook and app
├── requirements.txt
├── README.md
├── test_data.csv              # held-out test split (raw features + Target)
├── data/students_dropout.csv  # original UCI dataset
└── model/
    ├── train_models.ipynb     # EDA, preprocessing, training, evaluation
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── svm.joblib
    ├── metadata.json          # feature list, target name, label mapping
    └── metrics_comparison.csv
```

## d. Models used

All models are scikit-learn `Pipeline`s: `SimpleImputer` → `StandardScaler` →
classifier. They were trained on the same 80% training split and evaluated on the
same 20% test split.

| Model | Key settings |
|---|---|
| Logistic Regression | `max_iter=2000` |
| Decision Tree | `max_depth=6`, `min_samples_leaf=10` |
| KNN | `n_neighbors=15` |
| Naive Bayes | `GaussianNB` |
| Support Vector Machine | `kernel="rbf"`, `probability=True` |

### Comparison table (test set)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | _ | _ | _ | _ | _ | _ |
| Decision Tree | _ | _ | _ | _ | _ | _ |
| KNN | _ | _ | _ | _ | _ | _ |
| Naive Bayes | _ | _ | _ | _ | _ | _ |
| Support Vector Machine | _ | _ | _ | _ | _ | _ |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | _To be written from results_ |
| Decision Tree | _To be written from results_ |
| KNN | _To be written from results_ |
| Naive Bayes | _To be written from results_ |
| Support Vector Machine | _To be written from results_ |
| Overall winner | _To be written from results_ |

---

## Streamlit app features

- CSV upload of test data, plus a toggle to use the bundled `test_data.csv`
- Model selection dropdown (sidebar)
- The six evaluation metrics shown as metric cards
- Confusion-matrix heatmap and classification report
- A "Compare all models" tab with a metrics table and a grouped bar chart
- Download of the predictions as CSV
- Validation that reports any missing columns or unknown target labels

## How to run locally

```bash
# 1. create and activate a virtual environment (optional)
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

# 2. install dependencies
pip install -r requirements.txt jupyterlab

# 3. train the models (creates model/*.joblib, model/metadata.json, test_data.csv)
cd model
jupyter nbconvert --to notebook --execute --inplace train_models.ipynb
cd ..

# 4. launch the app
streamlit run app.py
```

## Deployment

The app is deployed on Streamlit Community Cloud from the `main` branch with
`app.py` as the entry point. The `scikit-learn` and `numpy` versions in
`requirements.txt` are pinned to the versions used for training, so the saved
`.joblib` files load correctly.
