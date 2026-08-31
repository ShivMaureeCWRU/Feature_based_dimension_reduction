import os
import pickle
import itertools
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import friedmanchisquare, wilcoxon

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


warnings.filterwarnings("ignore")


# Setup
REPRESENTATIONS_DIR="../representations"
OUTPUT_BASE_DIR="../statistical_tests"

os.makedirs(OUTPUT_BASE_DIR,exist_ok=True)



MODEL_TYPES=[
    "logistic_regression",
    "random_forest"
]


N_SPLITS=5
N_REPEATS=10
RANDOM_STATE=42



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


def holm_correction(p_values):
    m=len(p_values)
    sorted_indices=np.argsort(p_values)
    adjusted=np.empty(m)

    running_max=0

    for rank,idx in enumerate(sorted_indices):
        adjusted_p=(m-rank)*p_values[idx]
        running_max=max(running_max,adjusted_p)
        adjusted[idx]=min(running_max,1.0)

    return adjusted


def load_representations():
    X_dict={}
    labels_reference=None

    for representation_name,file_name in representation_files.items():
        path=os.path.join(REPRESENTATIONS_DIR,file_name)

        if not os.path.exists(path):
            print(f"Skipping {representation_name}: file not found.")
            continue

        data_dict=pickle.load(open(path,"rb"))

        X=np.asarray(data_dict["data"])
        labels=np.asarray(data_dict["labels"])

        if labels_reference is None:
            labels_reference=labels
        else:
            if not np.array_equal(labels_reference,labels):
                raise ValueError(
                    f"Labels for {representation_name} do not match the first representation."
                )

        X_dict[representation_name]=X

    return X_dict,labels_reference


def build_model(model_type,random_state):
    if model_type=="logistic_regression":
        model=Pipeline([
            ("scaler",StandardScaler()),
            ("logistic_regression",LogisticRegression(
                C=100,
                solver="lbfgs",
                max_iter=10000,
                random_state=random_state,
                n_jobs=-1
            ))
        ])

    elif model_type=="random_forest":
        model=RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",
            max_depth=None,
            min_samples_leaf=1,
            random_state=random_state,
            n_jobs=-1
        )

    else:
        raise ValueError("model_type must be 'logistic_regression' or 'random_forest'.")

    return model


def run_repeated_cv(X_dict,y,model_type):
    output_dir=os.path.join(OUTPUT_BASE_DIR,model_type)
    os.makedirs(output_dir,exist_ok=True)

    cv=RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE
    )

    fold_results=[]

    for fold_idx,(train_idx,test_idx) in enumerate(cv.split(next(iter(X_dict.values())),y),start=1):
        print(f"\n{model_type}: Fold {fold_idx}/{N_SPLITS*N_REPEATS}")

        for representation_name,X in X_dict.items():
            x_train=X[train_idx]
            x_test=X[test_idx]
            y_train=y[train_idx]
            y_test=y[test_idx]

            model=build_model(model_type,random_state=RANDOM_STATE+fold_idx)

            model.fit(x_train,y_train)
            y_pred=model.predict(x_test)

            accuracy=accuracy_score(y_test,y_pred)

            fold_results.append({
                "fold":fold_idx,
                "representation":representation_name,
                "accuracy":accuracy,
                "n_train":len(train_idx),
                "n_test":len(test_idx),
                "n_predictors":X.shape[1]
            })

            print(f"{representation_name}: {accuracy*100:.2f}%")

    fold_results_df=pd.DataFrame(fold_results)

    fold_results_df.to_csv(
        os.path.join(output_dir,"fold_level_accuracies.csv"),
        index=False
    )

    accuracy_matrix=fold_results_df.pivot(
        index="fold",
        columns="representation",
        values="accuracy"
    )

    accuracy_matrix.to_csv(
        os.path.join(output_dir,"accuracy_matrix.csv")
    )

    return accuracy_matrix,output_dir


def run_friedman_test(accuracy_matrix,output_dir):
    representation_names=list(accuracy_matrix.columns)

    friedman_stat,friedman_p=friedmanchisquare(
        *[accuracy_matrix[rep].values for rep in representation_names]
    )

    friedman_df=pd.DataFrame({
        "test":["Friedman test"],
        "statistic":[friedman_stat],
        "p_value":[friedman_p],
        "n_representations":[len(representation_names)],
        "n_folds":[accuracy_matrix.shape[0]]
    })

    friedman_df.to_csv(
        os.path.join(output_dir,"friedman_test_results.csv"),
        index=False
    )

    return friedman_stat,friedman_p


def run_pairwise_wilcoxon_tests(accuracy_matrix,output_dir):
    representation_names=list(accuracy_matrix.columns)

    pairwise_results=[]

    for rep_a,rep_b in itertools.combinations(representation_names,2):
        values_a=accuracy_matrix[rep_a].values
        values_b=accuracy_matrix[rep_b].values

        differences=values_a-values_b

        if np.allclose(differences,0):
            stat=np.nan
            p_value=1.0
        else:
            stat,p_value=wilcoxon(
                values_a,
                values_b,
                zero_method="pratt",
                alternative="two-sided"
            )

        mean_a=np.mean(values_a)
        mean_b=np.mean(values_b)

        pairwise_results.append({
            "representation_1":rep_a,
            "representation_2":rep_b,
            "mean_accuracy_1":mean_a,
            "mean_accuracy_2":mean_b,
            "mean_difference_1_minus_2":mean_a-mean_b,
            "wilcoxon_statistic":stat,
            "p_value":p_value
        })

    pairwise_df=pd.DataFrame(pairwise_results)

    pairwise_df["holm_adjusted_p_value"]=holm_correction(
        pairwise_df["p_value"].values
    )

    pairwise_df["significant_at_0.05"]=pairwise_df["holm_adjusted_p_value"]<0.05

    pairwise_df=pairwise_df.sort_values(
        by="holm_adjusted_p_value",
        ascending=True
    )

    pairwise_df.to_csv(
        os.path.join(output_dir,"pairwise_wilcoxon_results.csv"),
        index=False
    )

    return pairwise_df


def save_summary_table(accuracy_matrix,friedman_stat,friedman_p,output_dir):
    summary_df=pd.DataFrame({
        "representation":accuracy_matrix.columns,
        "mean_accuracy":[accuracy_matrix[col].mean() for col in accuracy_matrix.columns],
        "std_accuracy":[accuracy_matrix[col].std() for col in accuracy_matrix.columns],
        "min_accuracy":[accuracy_matrix[col].min() for col in accuracy_matrix.columns],
        "max_accuracy":[accuracy_matrix[col].max() for col in accuracy_matrix.columns]
    })

    summary_df=summary_df.sort_values(
        by="mean_accuracy",
        ascending=False
    )

    summary_df.to_csv(
        os.path.join(output_dir,"representation_accuracy_summary.csv"),
        index=False
    )

    text_path=os.path.join(output_dir,"statistical_test_summary.txt")

    with open(text_path,"w",encoding="utf-8") as f:
        f.write("Statistical Comparison of Feature Representations\n")

        f.write(f"Repeated stratified K-fold setup: {N_SPLITS} folds x {N_REPEATS} repeats\n")
        f.write(f"Total paired accuracy measurements: {N_SPLITS*N_REPEATS}\n\n")

        f.write("Friedman Test\n")
        f.write("-------------\n")
        f.write(f"Statistic: {friedman_stat:.6f}\n")
        f.write(f"p-value: {friedman_p:.6g}\n\n")

        if friedman_p<0.05:
            f.write("Conclusion: The Friedman test suggests that at least one representation differs significantly in accuracy.\n\n")
        else:
            f.write("Conclusion: The Friedman test does not provide evidence of a significant difference among representations.\n\n")

        f.write("Mean Accuracy by Representation\n")
        f.write("-------------------------------\n")

        for _,row in summary_df.iterrows():
            f.write(
                f"{row['representation']}: "
                f"mean={row['mean_accuracy']*100:.3f}%, "
                f"std={row['std_accuracy']*100:.3f}%, "
                f"min={row['min_accuracy']*100:.3f}%, "
                f"max={row['max_accuracy']*100:.3f}%\n"
            )

    return summary_df


def save_plots(accuracy_matrix,summary_df,output_dir,model_type):
    # Boxplot of fold accuracies
    plt.figure(figsize=(10,6))
    accuracy_matrix.boxplot(rot=45)
    plt.ylabel("Accuracy")
    plt.title(f"{model_type}: Fold-Level Accuracy by Representation")
    plt.ylim(0.98,1.0)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir,"fold_accuracy_boxplot.png"),
        dpi=300
    )
    plt.close()

    # Mean accuracy bar plot
    plot_df=summary_df.sort_values(by="mean_accuracy",ascending=False)

    plt.figure(figsize=(10,6))
    plt.bar(plot_df["representation"],plot_df["mean_accuracy"])
    plt.ylim(0.98,1.0)
    plt.xlabel("Feature Representation")
    plt.ylabel("Mean Repeated CV Accuracy")
    plt.title(f"{model_type}: Mean Accuracy Across Representations")
    plt.xticks(rotation=45,ha="right")

    for idx,value in enumerate(plot_df["mean_accuracy"]):
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
        os.path.join(output_dir,"mean_accuracy_barplot.png"),
        dpi=300
    )
    plt.close()


def main():
    X_dict,labels=load_representations()

    label_encoder=LabelEncoder()
    y=label_encoder.fit_transform(labels)

    for model_type in MODEL_TYPES:
        print(f"Statistical comparison for {model_type}")

        accuracy_matrix,output_dir=run_repeated_cv(
            X_dict,
            y,
            model_type
        )

        friedman_stat,friedman_p=run_friedman_test(
            accuracy_matrix,
            output_dir
        )

        pairwise_df=run_pairwise_wilcoxon_tests(
            accuracy_matrix,
            output_dir
        )

        summary_df=save_summary_table(
            accuracy_matrix,
            friedman_stat,
            friedman_p,
            output_dir
        )

        save_plots(
            accuracy_matrix,
            summary_df,
            output_dir,
            model_type
        )

        print("\nFriedman Test")
        print(f"Statistic: {friedman_stat:.6f}")
        print(f"p-value: {friedman_p:.6g}")

        print("\nTop pairwise Wilcoxon comparisons after Holm correction")
        print(pairwise_df.head(10))
        print(f"\nOutputs saved in: {output_dir}")


if __name__=="__main__":
    main()