import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from joblib import dump


OUTPUT_DIR = "../random_forest_outputs"

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


feature_names = []

for i in range(21):
    feature_names.append(f"x{i}")
    feature_names.append(f"y{i}")


rf_base_model = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)

param_grid = {
    "n_estimators": [100, 300, 500],
    "max_features": ["sqrt", "log2"],
    "max_depth": [None, 10, 20],
    "min_samples_leaf": [1, 2, 4]
}

grid_search = GridSearchCV(
    estimator=rf_base_model,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

grid_search.fit(x_train, y_train)

random_forest_model = grid_search.best_estimator_


tuning_results_df = pd.DataFrame(grid_search.cv_results_)
tuning_results_df.to_csv(
    os.path.join(OUTPUT_DIR, "random_forest_tuning_results.csv"),
    index=False
)

best_params_df = pd.DataFrame({
    "Hyperparameter": list(grid_search.best_params_.keys()),
    "Selected Value": list(grid_search.best_params_.values())
})

best_params_df.to_csv(
    os.path.join(OUTPUT_DIR, "random_forest_best_params.csv"),
    index=False
)


y_train_pred = random_forest_model.predict(x_train)
y_test_pred = random_forest_model.predict(x_test)

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

cm_df.to_csv(os.path.join(OUTPUT_DIR, "random_forest_confusion_matrix.csv"))


report_dict = classification_report(
    y_test,
    y_test_pred,
    target_names=class_names,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report_dict).T
report_df.to_csv(os.path.join(OUTPUT_DIR, "random_forest_classification_report.csv"))


importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": random_forest_model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
)

importance_df.to_csv(
    os.path.join(OUTPUT_DIR, "random_forest_feature_importances.csv"),
    index=False
)


summary_df = pd.DataFrame({
    "Metric": [
        "Training accuracy",
        "Training error",
        "Test accuracy",
        "Test error",
        "Cross-validation best accuracy",
        "Number of training observations",
        "Number of test observations",
        "Number of predictors",
        "Number of classes",
        "Number of trees",
        "Maximum tree depth",
        "Maximum features per split",
        "Minimum samples per leaf"
    ],
    "Value": [
        train_accuracy,
        train_error,
        test_accuracy,
        test_error,
        grid_search.best_score_,
        x_train.shape[0],
        x_test.shape[0],
        x_train.shape[1],
        len(class_names),
        random_forest_model.n_estimators,
        random_forest_model.max_depth,
        random_forest_model.max_features,
        random_forest_model.min_samples_leaf
    ]
})

summary_df.to_csv(os.path.join(OUTPUT_DIR, "random_forest_summary.csv"), index=False)


plt.figure(figsize=(10, 8))
plt.imshow(cm, aspect="auto")
plt.colorbar(label="Count")
plt.xticks(range(len(class_names)), class_names, rotation=90)
plt.yticks(range(len(class_names)), class_names)
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.title("Confusion Matrix for Random Forest Classifier")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "random_forest_confusion_matrix_plot.png"), dpi=300)
plt.close()


plt.figure(figsize=(6, 5))
plt.bar(["Training Accuracy", "Test Accuracy"], [train_accuracy, test_accuracy])
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Training and Test Accuracy")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "random_forest_accuracy_plot.png"), dpi=300)
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

plt.title("Random Forest Classification Report", pad=20)
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "random_forest_classification_report_plot.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()


summary_plot_df = summary_df.copy()

summary_plot_df["Value"] = summary_plot_df["Value"].apply(
    lambda x: round(x, 3) if isinstance(x, (float, np.floating)) else x
)

fig, ax = plt.subplots(figsize=(9, 5))
ax.axis("off")

table = ax.table(
    cellText=summary_plot_df.values,
    colLabels=summary_plot_df.columns,
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.1, 1.4)

plt.title("Random Forest Model Summary", pad=20)
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "random_forest_summary_plot.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()


fig, ax = plt.subplots(figsize=(7, 3))
ax.axis("off")

table = ax.table(
    cellText=best_params_df.values,
    colLabels=best_params_df.columns,
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.1, 1.5)

plt.title("Selected Random Forest Hyperparameters", pad=20)
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "random_forest_best_params_plot.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()


top_importance_df = importance_df.head(15)

plt.figure(figsize=(8, 6))
plt.barh(
    top_importance_df["feature"][::-1],
    top_importance_df["importance"][::-1]
)
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Top 15 Random Forest Feature Importances")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "random_forest_top_feature_importances.png"),
    dpi=300
)
plt.close()


plt.figure(figsize=(12, 5))
plt.bar(importance_df["feature"], importance_df["importance"])
plt.xlabel("Feature")
plt.ylabel("Feature Importance")
plt.title("Random Forest Feature Importances")
plt.xticks(rotation=90, fontsize=6)
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "random_forest_all_feature_importances.png"),
    dpi=300
)
plt.close()


print("\nRandom Forest Classifier Results")
print("--------------------------------")

print(f"\nNumber of observations: {data.shape[0]}")
print(f"Number of predictors: {data.shape[1]}")
print(f"Number of classes: {len(class_names)}")
print(f"Class labels: {list(class_names)}")

print("\nTrain/Test Split")
print("----------------")
print(f"Training observations: {x_train.shape[0]}")
print(f"Testing observations: {x_test.shape[0]}")

print("\nBest Hyperparameters")
print("--------------------")
print(grid_search.best_params_)
print(f"Best cross-validation accuracy: {grid_search.best_score_ * 100:.2f}%")

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

print("\nTop 15 Random Forest Feature Importances")
print("----------------------------------------")
print(top_importance_df)


dump(random_forest_model, "../models/random_forest_model.joblib")
dump(label_encoder, "../models/random_forest_label_encoder.joblib")

print("\nSaved random forest model and label encoder.")
print(f"\nAll diagnostic outputs saved in: {OUTPUT_DIR}")