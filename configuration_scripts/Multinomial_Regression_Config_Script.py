import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from joblib import dump


OUTPUT_DIR = "../logistic_outputs"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


data_dict = pickle.load(open("../data.pickle", "rb"))

data = np.asarray(data_dict["data"])
labels = np.asarray(data_dict["labels"])


label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

class_names = label_encoder.classes_


x_train, x_test, y_train, y_test = train_test_split(
    data,
    encoded_labels,
    test_size=0.2,
    shuffle=True,
    stratify=encoded_labels,
    random_state=42
)


scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


logistic_model = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs",
    max_iter=5000,
    random_state=42
)

logistic_model.fit(x_train_scaled, y_train)


y_train_pred = logistic_model.predict(x_train_scaled)
y_test_pred = logistic_model.predict(x_test_scaled)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

train_error = 1 - train_accuracy
test_error = 1 - test_accuracy


cm = confusion_matrix(y_test, y_test_pred)

cm_df = pd.DataFrame(
    cm,
    index=[f"Actual {name}" for name in class_names],
    columns=[f"Predicted {name}" for name in class_names]
)

cm_df.to_csv(os.path.join(OUTPUT_DIR, "logistic_confusion_matrix.csv"))


report_dict = classification_report(
    y_test,
    y_test_pred,
    target_names=class_names,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report_dict).T
report_df.to_csv(os.path.join(OUTPUT_DIR, "logistic_classification_report.csv"))


feature_names = []

for i in range(21):
    feature_names.append(f"x{i}")
    feature_names.append(f"y{i}")


coef_df = pd.DataFrame(
    logistic_model.coef_,
    index=[f"Class {name}" for name in class_names],
    columns=feature_names
)

coef_df.to_csv(os.path.join(OUTPUT_DIR, "logistic_coefficients.csv"))


top_coef_rows = []

for class_index, class_name in enumerate(class_names):
    class_coefs = logistic_model.coef_[class_index]
    top_indices = np.argsort(np.abs(class_coefs))[::-1][:5]

    for rank, feature_index in enumerate(top_indices, start=1):
        top_coef_rows.append({
            "class": class_name,
            "rank": rank,
            "feature": feature_names[feature_index],
            "coefficient": class_coefs[feature_index],
            "absolute_coefficient": abs(class_coefs[feature_index])
        })

top_coef_df = pd.DataFrame(top_coef_rows)
top_coef_df.to_csv(os.path.join(OUTPUT_DIR, "logistic_top_coefficients.csv"), index=False)


summary_df = pd.DataFrame({
    "Metric": [
        "Training accuracy",
        "Training error",
        "Test accuracy",
        "Test error",
        "Number of training observations",
        "Number of test observations",
        "Number of predictors",
        "Number of classes"
    ],
    "Value": [
        train_accuracy,
        train_error,
        test_accuracy,
        test_error,
        x_train.shape[0],
        x_test.shape[0],
        x_train.shape[1],
        len(class_names)
    ]
})

summary_df.to_csv(os.path.join(OUTPUT_DIR, "logistic_summary.csv"), index=False)


plt.figure(figsize=(10, 8))
plt.imshow(cm, aspect="auto")
plt.colorbar(label="Count")
plt.xticks(range(len(class_names)), class_names, rotation=90)
plt.yticks(range(len(class_names)), class_names)
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.title("Confusion Matrix for Multinomial Logistic Regression")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "logistic_confusion_matrix_plot.png"), dpi=300)
plt.close()


plt.figure(figsize=(6, 5))
plt.bar(["Training Accuracy", "Test Accuracy"], [train_accuracy, test_accuracy])
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Training and Test Accuracy")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "logistic_accuracy_plot.png"), dpi=300)
plt.close()


report_plot_df = report_df.copy()
report_plot_df = report_plot_df.round(3)

fig, ax = plt.subplots(figsize=(12, 6))
ax.axis("off")

table = ax.table(
    cellText=report_plot_df.values,
    rowLabels=report_plot_df.index,
    colLabels=report_plot_df.columns,
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.1, 1.4)

plt.title("Classification Report", pad=20)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "logistic_classification_report_plot.png"), dpi=300, bbox_inches="tight")
plt.close()


summary_plot_df = summary_df.copy()
summary_plot_df["Value"] = summary_plot_df["Value"].apply(
    lambda x: round(x, 3) if isinstance(x, (float, np.floating)) else x
)

fig, ax = plt.subplots(figsize=(8, 4))
ax.axis("off")

table = ax.table(
    cellText=summary_plot_df.values,
    colLabels=summary_plot_df.columns,
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.1, 1.5)

plt.title("Model Summary", pad=20)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "logistic_summary_plot.png"), dpi=300, bbox_inches="tight")
plt.close()


plt.figure(figsize=(14, 6))
plt.imshow(logistic_model.coef_, aspect="auto")
plt.colorbar(label="Coefficient Value")
plt.xticks(range(len(feature_names)), feature_names, rotation=90, fontsize=6)
plt.yticks(range(len(class_names)), class_names)
plt.xlabel("Feature")
plt.ylabel("Class")
plt.title("Logistic Regression Coefficient Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "logistic_coefficient_heatmap.png"), dpi=300)
plt.close()


selected_class = class_names[0]   # change this if you want another class
selected_index = list(class_names).index(selected_class)

selected_coefs = logistic_model.coef_[selected_index]
top_indices = np.argsort(np.abs(selected_coefs))[::-1][:10]

top_features = [feature_names[i] for i in top_indices]
top_values = [selected_coefs[i] for i in top_indices]

plt.figure(figsize=(8, 5))
plt.barh(top_features[::-1], top_values[::-1])
plt.xlabel("Coefficient Value")
plt.ylabel("Feature")
plt.title(f"Top 10 Coefficients for Class {selected_class}")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f"logistic_top_coefficients_class_{selected_class}.png"), dpi=300)
plt.close()


print("\nMultinomial Logistic Regression Results")
print("---------------------------------------")
print(f"\nNumber of observations: {data.shape[0]}")
print(f"Number of predictors: {data.shape[1]}")
print(f"Number of classes: {len(class_names)}")
print(f"Class labels: {list(class_names)}")

print("\nTrain/Test Split")
print("----------------")
print(f"Training observations: {x_train.shape[0]}")
print(f"Testing observations: {x_test.shape[0]}")

print("\nAccuracy Results")
print("----------------")
print(f"Training accuracy: {train_accuracy * 100:.2f}%")
print(f"Training error: {train_error * 100:.2f}%")
print(f"Test accuracy: {test_accuracy * 100:.2f}%")
print(f"Test error: {test_error * 100:.2f}%")

print("\nConfusion Matrix")
print("----------------")
print(cm_df)

print("\nClassification Report")
print("---------------------")
print(
    classification_report(
        y_test,
        y_test_pred,
        target_names=class_names,
        zero_division=0
    )
)

print("\nTop 5 Logistic Regression Coefficients by Absolute Value for Each Class")
print("-----------------------------------------------------------------------")
print(top_coef_df)


dump(logistic_model, "../models/multinomial_logistic_regression_model.joblib")
dump(scaler, "../models/logistic_regression_scaler.joblib")
dump(label_encoder, "../models/label_encoder.joblib")

print("\nSaved model, scaler, and label encoder.")
print(f"\nAll diagnostic outputs saved in: {OUTPUT_DIR}")