import pandas as pd
import numpy as np
import ast
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.multiclass import unique_labels
import plotly.graph_objects as go

# Load and preprocess data
df = pd.read_csv("../data/cleaned_dataset.csv")
df = df.dropna(subset=['BEST_MEDICATION', 'PRESENTING_PROBLEM'])


def safe_eval(val):
    try:
        return ast.literal_eval(val)
    except:
        return []


df['CONDITION'] = df['CONDITION'].apply(safe_eval)
df['ALLERGY'] = df['ALLERGY'].apply(safe_eval)
df['MEDICATION'] = df['MEDICATION'].apply(safe_eval)

label_encoder = LabelEncoder()
df['LABEL'] = label_encoder.fit_transform(df['BEST_MEDICATION'])

# Drop classes with <3 samples
value_counts = df['LABEL'].value_counts()
valid_labels = value_counts[value_counts >= 3].index
df = df[df['LABEL'].isin(valid_labels)]

# ✅ Re-label so classes are contiguous
df['LABEL'] = LabelEncoder().fit_transform(df['LABEL'])

# Feature encoding
features = df[['age', 'GENDER', 'RACE', 'ETHNICITY',
               'SMOKING_STATUS', 'PREGNANT', 'PRESENTING_PROBLEM']]
features_encoded = pd.get_dummies(features)
X = features_encoded
y = df['LABEL']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# RFECV
print("\n[INFO] Running RFECV...")
rfecv = RFECV(estimator=LogisticRegression(
    max_iter=1000, solver='liblinear'), step=1, cv=3, scoring='f1_weighted')
rfecv.fit(X_train_scaled, y_train)
X_train_scaled = rfecv.transform(X_train_scaled)
X_test_scaled = rfecv.transform(X_test_scaled)

# SMOTE
smote = SMOTE(random_state=42, k_neighbors=1)
X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)

# Tensor conversion
X_train_tensor = torch.tensor(X_train_bal, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_bal.values, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

# DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Neural network


class MediMatchNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MediMatchNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.model(x)


input_dim = X_train_tensor.shape[1]
output_dim = len(df['LABEL'].unique())
model = MediMatchNet(input_dim, output_dim)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 30
loss_history = []

print("\n[INFO] Training model...")
model.train()
for epoch in range(epochs):
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    loss_history.append(avg_loss)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# Save interactive Plotly loss curve
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=list(range(1, epochs + 1)),
    y=loss_history,
    mode='lines+markers',
    name='Loss',
    line=dict(color='royalblue', width=2),
    marker=dict(size=6)
))
fig.update_layout(
    title="Training Loss Over Epochs (Deep Learning Model)",
    xaxis_title="Epoch",
    yaxis_title="Loss",
    template="plotly_white",
    hovermode="x unified"
)
fig.write_html("training_loss_curve.html")
print("\n✅ Loss curve saved as training_loss_curve.html")

# Evaluation
print("\n[INFO] Evaluating model...")
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, preds = torch.max(outputs, 1)
    accuracy = accuracy_score(y_test_tensor, preds)
    print("\nModel: Deep Learning (Feedforward NN)")
    print("Accuracy:", accuracy)
    print("Classification Report:")
    print(classification_report(
        y_test_tensor, preds,
        labels=unique_labels(y_test_tensor, preds),
        target_names=label_encoder.inverse_transform(
            unique_labels(y_test_tensor, preds)),
        zero_division=0
    ))
