import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.feature_selection import RFECV
from imblearn.over_sampling import SMOTE
import pickle
import plotly.figure_factory as ff

# Load dataset
df = pd.read_csv("../data/cleaned_dataset.csv")
df = df.dropna(subset=['BEST_MEDICATION', 'PRESENTING_PROBLEM'])

# Convert stringified lists


def safe_eval(val):
    try:
        return ast.literal_eval(val)
    except:
        return []


df['CONDITION'] = df['CONDITION'].apply(safe_eval)
df['ALLERGY'] = df['ALLERGY'].apply(safe_eval)
df['MEDICATION'] = df['MEDICATION'].apply(safe_eval)

# Encode labels
label_encoder = LabelEncoder()
df['LABEL'] = label_encoder.fit_transform(df['BEST_MEDICATION'])

# Filter out classes with <3 samples
value_counts = df['LABEL'].value_counts()
valid_labels = value_counts[value_counts >= 3].index
df = df[df['LABEL'].isin(valid_labels)]

# Feature processing
features = df[['age', 'GENDER', 'RACE', 'ETHNICITY',
               'SMOKING_STATUS', 'PREGNANT', 'PRESENTING_PROBLEM']]
features_encoded = pd.get_dummies(features)
# Save full feature list BEFORE RFECV
with open("full_feature_columns.pkl", "wb") as f:
    pickle.dump(features_encoded.columns.tolist(), f)

X = features_encoded
y = df['LABEL']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# ---- Scale for logistic regression ----
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---- RFECV for Logistic Regression ----
print("\n[INFO] Running RFECV for Logistic Regression...")
logreg_for_rfe = LogisticRegression(max_iter=1000, solver='liblinear')
rfecv_lr = RFECV(estimator=logreg_for_rfe, step=1, cv=3,
                 scoring='f1_weighted', n_jobs=-1, verbose=0)
rfecv_lr.fit(X_train_scaled, y_train)

# Reduce to selected features
X_train_scaled = rfecv_lr.transform(X_train_scaled)
X_test_scaled = rfecv_lr.transform(X_test_scaled)

# Save RFECV feature mask
with open("selected_features_mask.pkl", "wb") as f:
    pickle.dump(rfecv_lr.support_, f)

# Save selected feature column names
selected_features = features_encoded.columns[rfecv_lr.support_]
with open("feature_columns.pkl", "wb") as f:
    pickle.dump(selected_features.tolist(), f)

# ---- RFECV for Random Forest ----
print("[INFO] Running RFECV for Random Forest...")
rf_for_rfe = RandomForestClassifier(n_estimators=100)
rfecv_rf = RFECV(estimator=rf_for_rfe, step=1, cv=3,
                 scoring='f1_weighted', n_jobs=-1, verbose=0)
rfecv_rf.fit(X_train, y_train)

# Reduce to selected features for RF
X_train = rfecv_rf.transform(X_train)
X_test = rfecv_rf.transform(X_test)

print(
    f"[INFO] Logistic Regression RFECV selected {X_train_scaled.shape[1]} features.")
print(f"[INFO] Random Forest RFECV selected {X_train.shape[1]} features.")

# ---- Apply SMOTE ----
smote = SMOTE(random_state=42, k_neighbors=1)
X_train_sm_scaled, y_train_sm_scaled = smote.fit_resample(
    X_train_scaled, y_train)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# ---- Hyperparameter Tuning ----

# Logistic Regression
logreg_params = {
    'C': [0.1, 1, 10],
    'solver': ['saga'],
    'penalty': ['l1', 'l2']
}
logreg_grid = GridSearchCV(
    LogisticRegression(class_weight='balanced', max_iter=20000, tol=1e-4),
    logreg_params,
    cv=3,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=1
)
logreg_grid.fit(X_train_sm_scaled, y_train_sm_scaled)
logreg_best = logreg_grid.best_estimator_

# Random Forest
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
}
rf_grid = GridSearchCV(
    RandomForestClassifier(class_weight='balanced'),
    rf_params,
    cv=3,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=1
)
rf_grid.fit(X_train_sm, y_train_sm)
rf_best = rf_grid.best_estimator_

# ---- Evaluation ----


def evaluate_model(name, y_true, y_pred):
    print(f"\nModel: {name}")
    print("Accuracy:", accuracy_score(y_true, y_pred))

    labels_in_test = unique_labels(y_true, y_pred)
    class_names = label_encoder.inverse_transform(labels_in_test).tolist()

    print("Classification Report:")
    print(classification_report(
        y_true, y_pred,
        labels=labels_in_test,
        target_names=class_names,
        zero_division=0
    ))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plotly heatmap
    fig = ff.create_annotated_heatmap(
        z=cm,
        x=class_names,
        y=class_names,
        colorscale='Blues',
        annotation_text=cm.astype(str),
        showscale=True
    )

    fig.update_layout(
        title=dict(
            text=f"Confusion Matrix - {name}",
            font=dict(size=20),
            x=0.5,
            xanchor='center',
            yanchor='top',
            pad=dict(t=40)
        ),
        width=2200,
        height=1200,
        margin=dict(l=300, r=300, t=200, b=300),
        xaxis=dict(
            tickangle=45,
            tickfont=dict(size=10),
            automargin=True
        ),
        yaxis=dict(
            tickfont=dict(size=10),
            automargin=True
        )
    )

    for ann in fig.layout.annotations:
        ann.font.size = 9

    html_filename = f"{name.replace(' ', '_')}_confusion_matrix.html"
    fig.write_html(html_filename)
    print(f"✅ Confusion matrix saved as {html_filename}")


# Predict and evaluate
logreg_preds = logreg_best.predict(X_test_scaled)
rf_preds = rf_best.predict(X_test)

evaluate_model("Logistic Regression (RFECV + SMOTE)", y_test, logreg_preds)
evaluate_model("Random Forest (RFECV + SMOTE)", y_test, rf_preds)

# Save models
with open("logreg_model.pkl", "wb") as f:
    pickle.dump(logreg_best, f)

with open("rf_model.pkl", "wb") as f:
    pickle.dump(rf_best, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
