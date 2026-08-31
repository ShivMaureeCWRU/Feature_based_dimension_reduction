import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from joblib import dump


REPRESENTATIONS_DIR="../kaggle_representations"
OUTPUT_BASE_DIR="../kaggle_logistic_regression_outputs"
MODEL_BASE_DIR="../models/kaggle_logistic_regression_models"

os.makedirs(OUTPUT_BASE_DIR,exist_ok=True)
os.makedirs(MODEL_BASE_DIR,exist_ok=True)

SUMMARY_LOG_PATH=os.path.join(
    OUTPUT_BASE_DIR,
    "kaggle_logistic_regression_console_summary.txt"
)

with open(SUMMARY_LOG_PATH,"w",encoding="utf-8") as f:
    f.write("Multinomial Logistic Regression Results Across Representations\n")


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


    print(f"Running Multinomial Logistic Regression for: {representation_name}")


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

    logistic_pipeline=Pipeline([
        ("scaler",StandardScaler()),
        ("logistic_regression",LogisticRegression(
            solver="lbfgs",
            max_iter=10000,
            random_state=42,
            n_jobs=-1
        ))
    ])

    param_grid={
        "logistic_regression__C":[0.001,0.01,0.1,1,10,100]
    }

    cv_strategy=StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    grid_search=GridSearchCV(
        estimator=logistic_pipeline,
        param_grid=param_grid,
        cv=cv_strategy,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(x_train,y_train)

    logistic_model=grid_search.best_estimator_
    final_logistic=logistic_model.named_steps["logistic_regression"]

    tuning_results_df=pd.DataFrame(grid_search.cv_results_)
    tuning_results_df.to_csv(
        os.path.join(OUTPUT_DIR,"kaggle_logistic_regression_tuning_results.csv"),
        index=False
    )

    best_params_df=pd.DataFrame({
        "Hyperparameter":list(grid_search.best_params_.keys()),
        "Selected Value":list(grid_search.best_params_.values())
    })

    best_params_df.to_csv(
        os.path.join(OUTPUT_DIR,"kaggle_logistic_regression_best_params.csv"),
        index=False
    )

    y_train_pred=logistic_model.predict(x_train)
    y_test_pred=logistic_model.predict(x_test)

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

    cm_df.to_csv(os.path.join(OUTPUT_DIR,"kaggle_logistic_regression_confusion_matrix.csv"))

    report_dict=classification_report(
        y_test,
        y_test_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    report_df=pd.DataFrame(report_dict).T
    report_df.to_csv(
        os.path.join(OUTPUT_DIR,"kaggle_logistic_regression_classification_report.csv")
    )

    coefficient_df=pd.DataFrame(
        final_logistic.coef_,
        index=class_names,
        columns=feature_names
    )

    coefficient_df.to_csv(
        os.path.join(OUTPUT_DIR,"kaggle_logistic_regression_coefficients.csv")
    )

    coef_long_df=coefficient_df.reset_index().melt(
        id_vars="index",
        var_name="feature",
        value_name="coefficient"
    )

    coef_long_df=coef_long_df.rename(columns={"index":"class"})
    coef_long_df["abs_coefficient"]=coef_long_df["coefficient"].abs()

    top_coef_df=coef_long_df.sort_values(
        by="abs_coefficient",
        ascending=False
    ).head(25)

    top_coef_df.to_csv(
        os.path.join(OUTPUT_DIR,"kaggle_logistic_regression_top_coefficients.csv"),
        index=False
    )

    top_by_class=[]

    for class_name in class_names:
        temp_df=coef_long_df[coef_long_df["class"]==class_name].copy()
        temp_df=temp_df.sort_values(by="abs_coefficient",ascending=False).head(10)
        top_by_class.append(temp_df)

    top_by_class_df=pd.concat(top_by_class,ignore_index=True)

    top_by_class_df.to_csv(
        os.path.join(OUTPUT_DIR,"kaggle_logistic_regression_top_coefficients_by_class.csv"),
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
            "Selected C",
            "Solver",
            "Maximum iterations"
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
            grid_search.best_params_["logistic_regression__C"],
            final_logistic.solver,
            final_logistic.max_iter
        ]
    })

    summary_df.to_csv(
        os.path.join(OUTPUT_DIR,"kaggle_logistic_regression_summary.csv"),
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
        "best_C":grid_search.best_params_["logistic_regression__C"]
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
        os.path.join(OUTPUT_DIR,"kaggle_logistic_regression_confusion_matrix_plot.png"),
        dpi=300
    )
    plt.close()

    plt.figure(figsize=(6,5))
    plt.bar(
        ["Training Accuracy","Test Accuracy","CV Best Accuracy"],
        [train_accuracy,test_accuracy,grid_search.best_score_]
    )
    plt.ylim(0,1)
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy: {representation_name}")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR,"kaggle_logistic_regression_accuracy_plot.png"),
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

    plt.title(f"Logistic Regression Classification Report: {representation_name}",pad=20)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR,"kaggle_logistic_regression_classification_report_plot.png"),
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

    plt.title(f"Logistic Regression Summary: {representation_name}",pad=20)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR,"kaggle_logistic_regression_summary_plot.png"),
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

    plt.title(f"Selected Logistic Regression Hyperparameters: {representation_name}",pad=20)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR,"kaggle_logistic_regression_best_params_plot.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    plt.figure(figsize=(12,7))
    plt.imshow(coefficient_df.values,aspect="auto")
    plt.colorbar(label="Coefficient")
    plt.yticks(range(len(class_names)),class_names)
    plt.xticks(range(len(feature_names)),feature_names,rotation=90,fontsize=5)
    plt.xlabel("Feature")
    plt.ylabel("Class")
    plt.title(f"Logistic Regression Coefficient Heatmap: {representation_name}")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR,"kaggle_logistic_regression_coefficient_heatmap.png"),
        dpi=300
    )
    plt.close()

    top_coef_plot_df=top_coef_df.copy()
    top_coef_plot_df["label"]=top_coef_plot_df["class"].astype(str)+": "+top_coef_plot_df["feature"]

    plt.figure(figsize=(9,7))
    plt.barh(
        top_coef_plot_df["label"][::-1],
        top_coef_plot_df["abs_coefficient"][::-1]
    )
    plt.xlabel("Absolute Coefficient")
    plt.ylabel("Class and Feature")
    plt.title(f"Top 25 Logistic Regression Coefficients: {representation_name}")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR,"kaggle_logistic_regression_top_coefficients.png"),
        dpi=300
    )
    plt.close()

    dump(
        logistic_model,
        os.path.join(MODEL_DIR,f"kaggle_logistic_regression_{representation_name}.joblib")
    )

    dump(
        label_encoder,
        os.path.join(MODEL_DIR,f"label_encoder_{representation_name}.joblib")
    )

    result_text=(
        "\nMultinomial Logistic Regression Results\n"
        "---------------------------------------\n"
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
    os.path.join(OUTPUT_BASE_DIR,"kaggle_logistic_regression_all_representations_summary.csv"),
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
    plt.ylim(0,1)
    plt.xlabel("Feature Representation")
    plt.ylabel("Test Accuracy")
    plt.title("Logistic Regression Test Accuracy Across Representations")
    plt.xticks(rotation=45,ha="right")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_BASE_DIR,"kaggle_logistic_regression_representation_test_accuracy.png"),
        dpi=300
    )
    plt.close()

    plt.figure(figsize=(10,6))
    plt.bar(
        all_results_df_sorted["representation"],
        all_results_df_sorted["cv_best_accuracy"]
    )
    plt.ylim(0,1)
    plt.xlabel("Feature Representation")
    plt.ylabel("Best 5-Fold CV Accuracy")
    plt.title("Logistic Regression Cross-Validation Accuracy Across Representations")
    plt.xticks(rotation=45,ha="right")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_BASE_DIR,"kaggle_logistic_regression_representation_cv_accuracy.png"),
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
            "best_C"
        ]
    ].copy()

    for col in ["train_accuracy","test_accuracy","cv_best_accuracy"]:
        summary_display_df[col]=summary_display_df[col].round(4)

    fig,ax=plt.subplots(figsize=(12,5))
    ax.axis("off")

    table=ax.table(
        cellText=summary_display_df.values,
        colLabels=summary_display_df.columns,
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0,1.4)

    plt.title("Logistic Regression Results Across Feature Representations",pad=20)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_BASE_DIR,"kaggle_logistic_regression_all_representations_summary_plot.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

print("\nFinished running Logistic Regression on all available representations.")
print(f"Combined summary saved in: {OUTPUT_BASE_DIR}")
print(f"Nice text summary saved to: {SUMMARY_LOG_PATH}")