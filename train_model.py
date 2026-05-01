import os
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import (
    train_model,
    inference,
    save_model,
    load_model,
    performance_on_categorical_slice,
    compute_model_metrics,
)

# Load the data
data = pd.read_csv("data/census.csv")

# Split the data into train and test
train, test = train_test_split(data, test_size=0.2, random_state=42)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Process the training data
X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True,
)

# Process the test data
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# Train the model
model = train_model(X_train, y_train)

# Save the model and encoder
save_model(model, "model/model.pkl")
save_model(encoder, "model/encoder.pkl")
print("Model saved to model/model.pkl")
print("Model saved to model/encoder.pkl")

# Load the model
model = load_model("model/model.pkl")
print("Loading model from model/model.pkl")

# Run inference on test data
preds = inference(model, X_test)

# Compute and print overall metrics
precision, recall, fbeta = compute_model_metrics(y_test, preds)
print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {fbeta:.4f}")

# Compute performance on slices and save to slice_output.txt
with open("slice_output.txt", "w") as f:
    for col in cat_features:
        for val in test[col].unique():
            count = len(test[test[col] == val])
            precision, recall, fbeta = performance_on_categorical_slice(
                test, col, val, cat_features, "salary", encoder, lb, model
            )
            f.write(f"{col}: {val}, Count: {count}\n")
            f.write(
                f"Precision: {precision:.4f} | "
                f"Recall: {recall:.4f} | "
                f"F1: {fbeta:.4f}\n"
            )
