import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from joblib import dump


REPRESENTATIONS_DIR="../kaggle_representations"
OUTPUT_BASE_DIR="../kaggle_random_forest_outputs"
MODEL_BASE_DIR="../models/kaggle_random_forest_models"


os.makedirs(OUTPUT_BASE_DIR,exist_ok=True)
os.makedirs(MODEL_BASE_DIR,exist_ok=True)

ACCURACY_Y_MIN=0.90
ACCURACY_Y_MAX=1.0

SUMMARY_LOG_PATH=os.path.join(
    OUTPUT_BASE_DIR,
    "kaggle_random_forest_console_summary.txt"
)

with open(SUMMARY_LOG_PATH,"w",encoding="utf-8") as f:
    f.write("Random Forest Results Across Representations\n")


representation_files={
    "raw_xy":"raw_xy.pickle",
    "raw_xyz":"raw_xyz.pickle",
    "translated_xy":"translated_xy.pickle",
    "scaled_xy":"scaled_xy.pickle",
    "distances_xy":"distances_xy.pickle",
    "normalized_distances_xy":"normalized_distances_xy.pickle",
    "angles_xy":"angles_xy.pickle",
    "hybrid":"hybrid.pickle"
}


def get_feature_names(representation_name,n_features):
    feature_names=[]

    if representation_name=="raw_xy":
        for i in range(21):
            feature_names.append(f"x{i}")
            feature_names.append(f"y{i}")

    elif representation_name=="raw_xyz":
        for i in range(21):
            feature_names.append(f"x{i}")
            feature_names.append(f"y{i}")
            feature_names.append(f"z{i}")

    elif representation_name=="translated_xy":
        for i in range(21):
            feature_names.append(f"x{i}_trans")
            feature_names.append(f"y{i}_trans")

    elif representation_name=="scaled_xy":
        for i in range(21):
            feature_names.append(f"x{i}_scale")
            feature_names.append(f"y{i}_scale")

    elif representation_name=="distances_xy":
        for a in range(21):
            for b in range(a+1,21):
                feature_names.append(f"d{a}_{b}")

    elif representation_name=="normalized_distances_xy":
        for a in range(21):
            for b in range(a+1,21):
                feature_names.append(f"norm_d{a}_{b}")

    elif representation_name=="angles_xy":
        angle_triples=[
            (0,1,2),(1,2,3),(2,3,4),
            (0,5,6),(5,6,7),(6,7,8),
            (0,9,10),(9,10,11),(10,11,12),
            (0,13,14),(13,14,15),(14,15,16),
            (0,17,18),(17,18,19),(18,19,20)
        ]

        for a,b,c in angle_triples:
            feature_names.append(f"theta_{a}_{b}_{c}")

    elif representation_name=="hybrid":
        for i in range(21):
            feature_names.append(f"x{i}_scale")
            feature_names.append(f"y{i}_scale")

        for a in range(21):
            for b in range(a+1,21):
                feature_names.append(f"norm_d{a}_{b}")

        angle_triples=[
            (0,1,2),(1,2,3),(2,3,4),
            (0,5,6),(5,6,7),(6,7,8),
            (0,9,10),(9,10,11),(10,11,12),
            (0,13,14),(13,14,15),(14,15,16),
            (0,17,18),(17,18,19),(18,19,20)
        ]

        for a,b,c in angle_triples:
            feature_names.append(f"theta_{a}_{b}_{c}")

    if len(feature_names)!=n_features:
        feature_names=[f"feature_{i}" for i in range(n_features)]

    return feature_names


all_results=[]


for representation_name,file_name in representation_files.items():

    print(f"Running Random Forest for: {representation_name}")

    representation_path=os.path.join(REPRESENTATIONS_DIR,file_name)

    if not os.path.exists(representation_path):
        skip_text=(
            f"\nSkipping {representation_name}: "
            f"file not found at {representation_path}\n"
        )

        print(skip_text)

        with open(SUMMARY_LOG_PATH,"a",encoding="utf-8") as f:
            f.write(skip_text)
            f.write("\n")

        continue

    OUTPUT_DIR=os.path.join(OUTPUT_BASE_DIR,representation_name)
    os.makedirs(OUTPUT_DIR,exist_ok=True)

    MODEL_DIR=os.path.join(MODEL_BASE_DIR,representation_name)
    os.makedirs(MODEL_DIR,exist_ok=True)

    data_dict=pickle.load(open(representation_path,"rb"))

    data=np.asarray(data_dict["data"])
    labels=np.asarray(data_dict["labels"])

    label_encoder=LabelEncoder()
    encoded_labels=label_encoder.fit_transform(labels)

    class_names=label_encoder.classes_

    x_train,x_test,y_train,y_test=train_test_split(
        data,
        encoded_labels,
        test_size=0.2,
        shuffle=True,
        stratify=encoded_labels,
        random_state=42
    )

    feature_names=get_feature_names(representation_name,data.shape[1])

    rf_base_model=RandomForestClassifier(
        random_state=42,
        n_jobs=-1
    )

    param_grid={
        "n_estimators":[100,300,500],
        "max_features":["sqrt","log2"],
        "max_depth":[None,10,20],
        "min_samples_leaf":[1,2,4]
    }

    cv_strategy=StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    grid_search=GridSearchCV(
        estimator=rf_base_model,
        param_grid=param_grid,
        cv=cv_strategy,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(x_train,y_train)

    random_forest_model=grid_search.best_estimator_

    tuning_results_df=pd.DataFrame(grid_search.cv_results_)
    tuning_results_df.to_csv(
        os.path.join(OUTPUT_DIR,"kaggle_random_forest_tuning_results.csv"),
        index=False
    )

    best_params_df=pd.DataFrame({
        "Hyperparameter":list(grid_search.best_params_.keys()),
        "Selected Value":list(grid_search.best_params_.values())
    })

    best_params_df.to_csv(
        os.path.join(OUTPUT_DIR,"kaggle_random_forest_best_params.csv"),
        index=False
    )

    y_train_pred=random_forest_model.predict(x_train)
    y_test_pred=random_forest_model.predict(x_test)

    train_accuracy=accuracy_score(y_train,y_train_pred)
    test_accuracy=accuracy_score(y_test,y_test_pred)

    train_error=1-train_accuracy
    test_error=1-test_accuracy

    cm=confusion_matrix(y_test,y_test_pred)

    cm_df=pd.DataFrame(
        cm,
        index=[f"Actual {name}" for name in class_names],
        columns=[f"Predicted {name}" for name in class_names]
    )

    cm_df.to_csv(os.path.join(OUTPUT_DIR,"kaggle_random_forest_confusion_matrix.csv"))

    report_dict=classification_report(
        y_test,
        y_test_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    report_df=pd.DataFrame(report_dict).T
    report_df.to_csv(
        os.path.join(OUTPUT_DIR,"kaggle_random_forest_classification_report.csv")
    )

    importance_df=pd.DataFrame({
        "feature":feature_names,
        "importance":random_forest_model.feature_importances_
    })

    importance_df=importance_df.sort_values(
        by="importance",
        ascending=False
    )

    importance_df.to_csv(
        os.path.join(OUTPUT_DIR,"kaggle_random_forest_feature_importances.csv"),
        index=False
    )

    summary_df=pd.DataFrame({
        "Metric":[
            "Representation",
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
        "Value":[
            representation_name,
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

    summary_df.to_csv(
        os.path.join(OUTPUT_DIR,"kaggle_random_forest_summary.csv"),
        index=False
    )

    all_results.append({
        "representation":representation_name,
        "n_observations":data.shape[0],
        "n_predictors":data.shape[1],
        "n_classes":len(class_names),
        "train_accuracy":train_accuracy,
        "test_accuracy":test_accuracy,
        "train_error":train_error,
        "test_error":test_error,
        "cv_best_accuracy":grid_search.best_score_,
        "best_n_estimators":random_forest_model.n_estimators,
        "best_max_depth":random_forest_model.max_depth,
        "best_max_features":random_forest_model.max_features,
        "best_min_samples_leaf":random_forest_model.min_samples_leaf
    })

    plt.figure(figsize=(10,8))
    plt.imshow(cm,aspect="auto")
    plt.colorbar(label="Count")
    plt.xticks(range(len(class_names)),class_names,rotation=90)
    plt.yticks(range(len(class_names)),class_names)
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.title(f"Confusion Matrix: {representation_name}")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j,i,str(cm[i,j]),ha="center",va="center",fontsize=8)

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR,"kaggle_random_forest_confusion_matrix_plot.png"),
        dpi=300
    )
    plt.close()

    plt.figure(figsize=(6,5))
    plt.bar(
        ["Training Accuracy","Test Accuracy","CV Best Accuracy"],
        [train_accuracy,test_accuracy,grid_search.best_score_]
    )
    plt.ylim(ACCURACY_Y_MIN,ACCURACY_Y_MAX)
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy: {representation_name}")
    plt.xticks(rotation=20)

    for idx,value in enumerate([train_accuracy,test_accuracy,grid_search.best_score_]):
        plt.text(
            idx,
            value,
            f"{value*100:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR,"kaggle_random_forest_accuracy_plot.png"),
        dpi=300
    )
    plt.close()

    report_plot_df=report_df.copy()
    report_plot_df=report_plot_df.round(3)

    fig,ax=plt.subplots(figsize=(12,6))
    ax.axis("off")

    table=ax.table(
        cellText=report_plot_df.values,
        rowLabels=report_plot_df.index,
        colLabels=report_plot_df.columns,
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.1,1.4)

    plt.title(f"Random Forest Classification Report: {representation_name}",pad=20)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR,"kaggle_random_forest_classification_report_plot.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    summary_plot_df=summary_df.copy()

    summary_plot_df["Value"]=summary_plot_df["Value"].apply(
        lambda x: round(x,3) if isinstance(x,(float,np.floating)) else x
    )

    fig,ax=plt.subplots(figsize=(9,5))
    ax.axis("off")

    table=ax.table(
        cellText=summary_plot_df.values,
        colLabels=summary_plot_df.columns,
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1,1.4)

    plt.title(f"Random Forest Summary: {representation_name}",pad=20)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR,"kaggle_random_forest_summary_plot.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    fig,ax=plt.subplots(figsize=(7,3))
    ax.axis("off")

    table=ax.table(
        cellText=best_params_df.values,
        colLabels=best_params_df.columns,
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1,1.5)

    plt.title(f"Selected Hyperparameters: {representation_name}",pad=20)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR,"kaggle_random_forest_best_params_plot.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    top_importance_df=importance_df.head(15)

    plt.figure(figsize=(8,6))
    plt.barh(
        top_importance_df["feature"][::-1],
        top_importance_df["importance"][::-1]
    )
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.title(f"Top 15 Feature Importances: {representation_name}")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR,"kaggle_random_forest_top_feature_importances.png"),
        dpi=300
    )
    plt.close()

    plt.figure(figsize=(12,5))
    plt.bar(importance_df["feature"],importance_df["importance"])
    plt.xlabel("Feature")
    plt.ylabel("Feature Importance")
    plt.title(f"Feature Importances: {representation_name}")
    plt.xticks(rotation=90,fontsize=6)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR,"kaggle_random_forest_all_feature_importances.png"),
        dpi=300
    )
    plt.close()

    dump(
        random_forest_model,
        os.path.join(MODEL_DIR,f"kaggle_random_forest_{representation_name}.joblib")
    )

    dump(
        label_encoder,
        os.path.join(MODEL_DIR,f"label_encoder_{representation_name}.joblib")
    )

    result_text=(
        "\nRandom Forest Classifier Results\n"
        "--------------------------------\n"
        f"Representation: {representation_name}\n"
        f"Number of observations: {data.shape[0]}\n"
        f"Number of predictors: {data.shape[1]}\n"
        f"Number of classes: {len(class_names)}\n"
        f"Best hyperparameters: {grid_search.best_params_}\n"
        f"Best cross-validation accuracy: {grid_search.best_score_ * 100:.2f}%\n"
        f"Training accuracy: {train_accuracy * 100:.2f}%\n"
        f"Test accuracy: {test_accuracy * 100:.2f}%\n"
        f"Outputs saved in: {OUTPUT_DIR}\n"
    )

    print(result_text)

    with open(SUMMARY_LOG_PATH,"a",encoding="utf-8") as f:
        f.write(result_text)
        f.write("\n")


all_results_df=pd.DataFrame(all_results)

all_results_df.to_csv(
    os.path.join(OUTPUT_BASE_DIR,"kaggle_random_forest_all_representations_summary.csv"),
    index=False
)

if len(all_results_df)>0:
    all_results_df_sorted=all_results_df.sort_values(
        by="test_accuracy",
        ascending=False
    )

    plt.figure(figsize=(10,6))
    plt.bar(
        all_results_df_sorted["representation"],
        all_results_df_sorted["test_accuracy"]
    )
    plt.ylim(ACCURACY_Y_MIN,ACCURACY_Y_MAX)
    plt.xlabel("Feature Representation")
    plt.ylabel("Test Accuracy")
    plt.title("Random Forest Test Accuracy Across Representations")
    plt.xticks(rotation=45,ha="right")

    for idx,value in enumerate(all_results_df_sorted["test_accuracy"]):
        plt.text(
            idx,
            value,
            f"{value*100:.2f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90
        )

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_BASE_DIR,"kaggle_random_forest_representation_test_accuracy.png"),
        dpi=300
    )
    plt.close()

    plt.figure(figsize=(10,6))
    plt.bar(
        all_results_df_sorted["representation"],
        all_results_df_sorted["cv_best_accuracy"]
    )
    plt.ylim(ACCURACY_Y_MIN,ACCURACY_Y_MAX)
    plt.xlabel("Feature Representation")
    plt.ylabel("Best 5-Fold CV Accuracy")
    plt.title("Random Forest Cross-Validation Accuracy Across Representations")
    plt.xticks(rotation=45,ha="right")

    for idx,value in enumerate(all_results_df_sorted["cv_best_accuracy"]):
        plt.text(
            idx,
            value,
            f"{value*100:.2f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90
        )

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_BASE_DIR,"kaggle_random_forest_representation_cv_accuracy.png"),
        dpi=300
    )
    plt.close()

    summary_display_df=all_results_df_sorted[
        [
            "representation",
            "n_predictors",
            "train_accuracy",
            "test_accuracy",
            "cv_best_accuracy",
            "best_n_estimators",
            "best_max_depth",
            "best_max_features",
            "best_min_samples_leaf"
        ]
    ].copy()

    for col in ["train_accuracy","test_accuracy","cv_best_accuracy"]:
        summary_display_df[col]=summary_display_df[col].round(4)

    fig,ax=plt.subplots(figsize=(14,5))
    ax.axis("off")

    table=ax.table(
        cellText=summary_display_df.values,
        colLabels=summary_display_df.columns,
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0,1.4)

    plt.title("Random Forest Results Across Feature Representations",pad=20)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_BASE_DIR,"kaggle_random_forest_all_representations_summary_plot.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

print("\nFinished running Random Forest on all available representations.")
print(f"Combined summary saved in: {OUTPUT_BASE_DIR}")
print(f"Nice text summary saved to: {SUMMARY_LOG_PATH}")