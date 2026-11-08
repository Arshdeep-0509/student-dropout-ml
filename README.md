# Student Dropout & Academic Success — ML Classification

**Live Streamlit app:** https://student-dropout-ml-rl9tca3dijal4v68ecaorr.streamlit.app/

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

https://github.com/Arshdeep-0509/student-dropout-ml

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
| Logistic Regression | 0.7684 | 0.8961 | 0.7500 | 0.7684 | 0.7531 | 0.6150 |
| Decision Tree | 0.7458 | 0.8687 | 0.7338 | 0.7458 | 0.7350 | 0.5780 |
| KNN | 0.7028 | 0.8368 | 0.6855 | 0.7028 | 0.6755 | 0.5019 |
| Naive Bayes | 0.6588 | 0.8118 | 0.6334 | 0.6588 | 0.6417 | 0.4279 |
| Support Vector Machine | 0.7582 | 0.8865 | 0.7465 | 0.7582 | 0.7454 | 0.5986 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Top scorer on all six metrics (Accuracy 0.7684, AUC 0.8961, MCC 0.6150). Train accuracy is 0.7731, only 0.005 above test, so it generalises well and hardly overfits. Strong results from a linear model suggest the classes are mostly separable by a weighted sum of the scaled features, especially the 1st/2nd semester approved units and grades. Its weak spot is the `Enrolled` class (recall 0.33): many of these students get predicted as Graduate. |
| Decision Tree | Third overall (Accuracy 0.7458, F1 0.7350). Limiting depth to 6 kept the train–test gap small (0.783 vs 0.746). It has the best `Enrolled` recall of the five models (0.38), because its rule-based splits can isolate small groups. Its AUC (0.8687) is lower than LR/SVM, as a shallow tree produces coarse, step-like probabilities. |
| KNN | Fourth (Accuracy 0.7028, MCC 0.5019). With 36 features, distances between students become less meaningful, and the majority `Graduate` class dominates the 15 neighbours. As a result, `Enrolled` recall falls to 0.21 while `Graduate` recall is 0.93. It is also the largest saved model (~1 MB), since it has to store the whole training set. |
| Naive Bayes | Weakest model (Accuracy 0.6588, F1 0.6417, MCC 0.4279). GaussianNB assumes features are independent and normally distributed, but this dataset has many strongly correlated pairs (e.g. 1st vs 2nd semester units, mother's vs father's qualification) and many integer-coded categorical columns. It catches only 18% of `Enrolled` students. Its one advantage is speed: training takes around 0.01 s. |
| Support Vector Machine | Second best (Accuracy 0.7582, AUC 0.8865, F1 0.7454), close behind Logistic Regression, and it has the highest `Dropout` precision (0.84). It shows the largest overfitting gap (train 0.830 vs test 0.758): the RBF kernel fits the training data more tightly without improving on unseen students. It is also the slowest to train (~2 s with probability estimates). |
| Overall winner | **Logistic Regression.** It has the best accuracy, AUC, F1 and MCC, with almost no overfitting, and it is the simplest and most interpretable model. SVM is a close second. All models find `Graduate` and `Dropout` much easier than `Enrolled`, the minority class (18% of rows), which sits between the other two outcomes. Class weighting or oversampling would be the next step to improve it. |

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
