from pathlib import Path
import re
import pickle
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score




PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "final_results_eda"
OUTPUT_DIR.mkdir(exist_ok=True)


HAGRID_LANDMARK_FILE = None
HAGRID_LABEL_FILE = None


# Experiment settings

RANDOM_SEED = 42

N_SPLITS = 5
N_REPEATS = 10

INNER_CV_FOLDS = 3

N_JOBS = -1

CLASSIFIER_ORDER = [
    "Multinomial logistic",
    "Random forest",
]

ABLATION_ORDER = [
    "Hybrid",
    "Hybrid - scale",
    "Hybrid - norm-dist",
    "Hybrid - angle",
]

warnings.filterwarnings("ignore")


# Flexible loading

LABEL_CANDIDATES = [
    "label", "labels",
    "class", "classes",
    "gesture", "gestures",
    "target", "targets",
    "y",
    "category",
    "sign",
]

FEATURE_CANDIDATES = [
    "X",
    "x",
    "data",
    "features",
    "landmarks",
    "raw_xy",
    "raw_xyz",
    "coordinates",
]


def read_pickle_any(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def read_any(path):
    path = Path(path)

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    if path.suffix.lower() in [".pkl", ".pickle"]:
        return read_pickle_any(path)

    if path.suffix.lower() == ".npy":
        return np.load(path, allow_pickle=True)

    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)

    raise ValueError(f"Unsupported file type: {path}")


def find_label_column(df):
    lower_cols = {str(c).lower(): c for c in df.columns}

    for candidate in LABEL_CANDIDATES:
        if candidate in lower_cols:
            return lower_cols[candidate]

    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            return col

    raise ValueError("Could not identify label column.")


def extract_xy_columns_by_name(df):
    cols = list(df.columns)

    patterns = [
        (r"^x[_\-]?(\d+)$", r"^y[_\-]?(\d+)$"),
        (r"^(\d+)[_\-]?x$", r"^(\d+)[_\-]?y$"),
        (r"^landmark[_\-]?(\d+)[_\-]?x$", r"^landmark[_\-]?(\d+)[_\-]?y$"),
        (r"^lm[_\-]?(\d+)[_\-]?x$", r"^lm[_\-]?(\d+)[_\-]?y$"),
    ]

    for x_pat, y_pat in patterns:
        x_map = {}
        y_map = {}

        for col in cols:
            col_str = str(col).lower()
            mx = re.match(x_pat, col_str)
            my = re.match(y_pat, col_str)

            if mx:
                x_map[int(mx.group(1))] = col

            if my:
                y_map[int(my.group(1))] = col

        if set(x_map.keys()) == set(range(21)) and set(y_map.keys()) == set(range(21)):
            ordered = []

            for j in range(21):
                ordered.extend([x_map[j], y_map[j]])

            return ordered

    return None


def array_to_landmarks_and_labels(X, y):
    X = np.asarray(X)
    y = np.asarray(y).astype(str)

    if X.ndim == 3 and X.shape[1:] == (21, 2):
        L = X.astype(float)

    elif X.ndim == 3 and X.shape[1:] == (21, 3):
        L = X[:, :, :2].astype(float)

    elif X.ndim == 2 and X.shape[1] == 42:
        L = X.astype(float).reshape(-1, 21, 2)

    elif X.ndim == 2 and X.shape[1] == 63:
        X3 = X.astype(float).reshape(-1, 21, 3)
        L = X3[:, :, :2]

    else:
        raise ValueError(f"Could not convert feature array with shape {X.shape} to landmarks.")

    if len(L) != len(y):
        raise ValueError(f"Landmark rows and labels do not match: {len(L)} vs {len(y)}")

    return L, y


def dataframe_to_landmarks_and_labels(df):
    label_col = find_label_column(df)
    xy_cols = extract_xy_columns_by_name(df)

    if xy_cols is None:
        numeric_cols = [
            col for col in df.columns
            if col != label_col and pd.api.types.is_numeric_dtype(df[col])
        ]

        if len(numeric_cols) < 42:
            raise ValueError(f"Need at least 42 numeric landmark columns. Found {len(numeric_cols)}.")

        xy_cols = numeric_cols[:42]

    X = df[xy_cols].to_numpy(dtype=float)
    y = df[label_col].astype(str).to_numpy()

    return array_to_landmarks_and_labels(X, y)


def load_label_file(label_file, expected_n):
    obj = read_any(label_file)

    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] == 1:
            y = obj.iloc[:, 0].to_numpy()
        else:
            label_col = find_label_column(obj)
            y = obj[label_col].to_numpy()
    else:
        y = np.asarray(obj)

    y = np.asarray(y).astype(str)

    if len(y) != expected_n:
        raise ValueError(f"Label length mismatch: expected {expected_n}, got {len(y)}.")

    return y


def find_labels_nearby(source_path, expected_n):
    if HAGRID_LABEL_FILE is not None:
        return load_label_file(HAGRID_LABEL_FILE, expected_n)

    source_path = Path(source_path)

    search_dirs = [
        source_path.parent,
        PROJECT_ROOT / "hagrid_100",
        PROJECT_ROOT / "HaGRID",
        PROJECT_ROOT / "hagrid",
        PROJECT_ROOT / "Representations",
        PROJECT_ROOT / "representations",
        PROJECT_ROOT / "data",
    ]

    label_names = [
        "labels",
        "label",
        "y",
        "targets",
        "target",
        "classes",
        "class",
        "gestures",
        "gesture",
        "hagrid_labels",
        "hagrid_100_labels",
    ]

    for folder in search_dirs:
        if not folder.exists():
            continue

        for label_name in label_names:
            for ext in [".pickle", ".pkl", ".csv", ".npy"]:
                candidate = folder / f"{label_name}{ext}"

                if not candidate.exists():
                    continue

                try:
                    y = load_label_file(candidate, expected_n)
                    print(f"Loaded labels from: {candidate}")
                    return y
                except Exception:
                    pass

    raise ValueError(
        f"The feature file {source_path} appears to contain X only. "
        f"No matching label file was found. Set HAGRID_LABEL_FILE at the top."
    )


def object_to_landmarks_and_labels(obj, source_path=None):
    if isinstance(obj, pd.DataFrame):
        return dataframe_to_landmarks_and_labels(obj)

    if isinstance(obj, dict):
        keys_lower = {str(k).lower(): k for k in obj.keys()}

        X = None
        y = None

        for key in FEATURE_CANDIDATES:
            if key.lower() in keys_lower:
                X = obj[keys_lower[key.lower()]]
                break

        for key in LABEL_CANDIDATES:
            if key.lower() in keys_lower:
                y = obj[keys_lower[key.lower()]]
                break

        if X is None:
            for _, v in obj.items():
                arr = np.asarray(v)

                if (
                    (arr.ndim == 2 and arr.shape[1] in [42, 63]) or
                    (arr.ndim == 3 and arr.shape[1] == 21 and arr.shape[2] in [2, 3])
                ):
                    X = v
                    break

        if y is None and X is not None:
            expected_n = len(np.asarray(X))

            for _, v in obj.items():
                arr = np.asarray(v)

                if arr.ndim == 1 and len(arr) == expected_n:
                    y = v
                    break

        if X is None:
            raise ValueError(f"Could not extract features from dict. Keys: {list(obj.keys())}")

        if y is None:
            y = find_labels_nearby(source_path, expected_n=len(np.asarray(X)))

        return array_to_landmarks_and_labels(X, y)

    if isinstance(obj, (tuple, list)):
        if len(obj) >= 2:
            return array_to_landmarks_and_labels(obj[0], obj[1])

        if len(obj) == 1:
            return object_to_landmarks_and_labels(obj[0], source_path=source_path)

    if isinstance(obj, np.ndarray):
        y = find_labels_nearby(source_path, expected_n=len(obj))
        return array_to_landmarks_and_labels(obj, y)

    raise ValueError(f"Unsupported object type: {type(obj)}")


def candidate_hagrid_landmark_files():
    candidate_names = [
        "hagrid_raw_xy",
        "hagrid_100_raw_xy",
        "raw_xy",
        "raw_xyz",
        "hagrid_landmarks",
        "hagrid_100_landmarks",
        "landmarks",
        "features",
    ]

    candidate_dirs = [
        PROJECT_ROOT / "hagrid_100",
        PROJECT_ROOT / "HaGRID",
        PROJECT_ROOT / "hagrid",
        PROJECT_ROOT / "hagrid_100_representations",
        PROJECT_ROOT / "Representations" / "HaGRID",
        PROJECT_ROOT / "Representations" / "hagrid",
        PROJECT_ROOT / "representations" / "HaGRID",
        PROJECT_ROOT / "representations" / "hagrid",
        PROJECT_ROOT / "Representations",
        PROJECT_ROOT / "representations",
        PROJECT_ROOT / "data",
    ]

    candidates = []

    for folder in candidate_dirs:
        for name in candidate_names:
            for ext in [".pickle", ".pkl", ".csv", ".npy", ".parquet"]:
                candidates.append(folder / f"{name}{ext}")

    for pattern in [
        "*hagrid*raw*xy*.pickle",
        "*hagrid*raw*xy*.pkl",
        "*hagrid*landmark*.pickle",
        "*hagrid*landmark*.pkl",
        "*hagrid*landmark*.csv",
        "*hagrid*features*.csv",
    ]:
        candidates.extend(PROJECT_ROOT.glob(pattern))

        if (PROJECT_ROOT / "final_eda").exists():
            candidates.extend((PROJECT_ROOT / "final_eda").glob(pattern))

        if (PROJECT_ROOT / "final_results_eda").exists():
            candidates.extend((PROJECT_ROOT / "final_results_eda").glob(pattern))

    unique = []
    seen = set()

    for path in candidates:
        path = Path(path)

        if str(path) not in seen:
            unique.append(path)
            seen.add(str(path))

    return unique


def load_hagrid_landmarks():
    if HAGRID_LANDMARK_FILE is not None:
        candidates = [Path(HAGRID_LANDMARK_FILE)]
    else:
        candidates = candidate_hagrid_landmark_files()

    attempted = []

    for path in candidates:
        if not path.exists():
            continue

        try:
            obj = read_any(path)
            L, y = object_to_landmarks_and_labels(obj, source_path=path)

            print(f"Loaded HaGRID landmarks from: {path}")
            print(f"Landmarks shape: {L.shape}")
            print(f"Labels shape: {y.shape}")
            print(f"Classes: {len(np.unique(y))}")

            return L, y

        except Exception as e:
            attempted.append((path, str(e)))

    print("\nCandidate files tried:")

    for path, error in attempted:
        print(f"  {path} -> {error}")

    raise FileNotFoundError(
        "Could not auto-detect HaGRID landmark file. "
        "Set HAGRID_LANDMARK_FILE and, if needed, HAGRID_LABEL_FILE at the top of the script."
    )


# Feature components

EPS = 1e-8

PAIRWISE_PAIRS = [(i, j) for i in range(21) for j in range(i + 1, 21)]

ANGLE_TRIPLES = [
    (0, 1, 2), (1, 2, 3), (2, 3, 4),
    (0, 5, 6), (5, 6, 7), (6, 7, 8),
    (0, 9, 10), (9, 10, 11), (10, 11, 12),
    (0, 13, 14), (13, 14, 15), (14, 15, 16),
    (0, 17, 18), (17, 18, 19), (18, 19, 20),
]


def phi_scaled(L):
    mins = L.min(axis=1, keepdims=True)
    maxs = L.max(axis=1, keepdims=True)
    ranges = np.maximum(maxs - mins, EPS)

    return ((L - mins) / ranges).reshape(L.shape[0], -1)


def phi_pairwise_distance(L):
    feats = []

    for a, b in PAIRWISE_PAIRS:
        feats.append(np.linalg.norm(L[:, a, :] - L[:, b, :], axis=1))

    return np.column_stack(feats)


def phi_norm_distance(L):
    D = phi_pairwise_distance(L)
    scale = np.maximum(D.max(axis=1, keepdims=True), EPS)

    return D / scale


def phi_angles(L):
    feats = []

    for a, b, c in ANGLE_TRIPLES:
        u = L[:, a, :] - L[:, b, :]
        v = L[:, c, :] - L[:, b, :]

        dot = np.sum(u * v, axis=1)
        denom = np.maximum(np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1), EPS)

        cosang = np.clip(dot / denom, -1.0, 1.0)
        feats.append(np.arccos(cosang))

    return np.column_stack(feats)


def build_ablation_features(L):
    scale = phi_scaled(L)
    norm_dist = phi_norm_distance(L)
    angle = phi_angles(L)

    features = {
        "Hybrid": np.concatenate([scale, norm_dist, angle], axis=1),
        "Hybrid - scale": np.concatenate([norm_dist, angle], axis=1),
        "Hybrid - norm-dist": np.concatenate([scale, angle], axis=1),
        "Hybrid - angle": np.concatenate([scale, norm_dist], axis=1),
    }

    dimensions = {
        name: X.shape[1]
        for name, X in features.items()
    }

    return features, dimensions


# Models


def make_classifier(classifier_name):
    if classifier_name == "Multinomial logistic":
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=5000,
                        solver="lbfgs",
                        multi_class="auto",
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        )

        param_grid = {
            "model__C": [0.001, 0.01, 0.1, 1, 10, 100],
        }

        return pipe, param_grid

    if classifier_name == "Random forest":
        pipe = Pipeline(
            [
                (
                    "model",
                    RandomForestClassifier(
                        random_state=RANDOM_SEED,
                        n_jobs=N_JOBS,
                        class_weight=None,
                    ),
                ),
            ]
        )

        param_grid = {
            "model__n_estimators": [300],
            "model__max_depth": [None, 20, 40],
            "model__min_samples_leaf": [1, 2],
            "model__max_features": ["sqrt", "log2"],
        }

        return pipe, param_grid

    raise ValueError(f"Unknown classifier: {classifier_name}")


def fit_tuned_model(classifier_name, X_train, y_train, fold_seed):
    pipe, param_grid = make_classifier(classifier_name)

    inner_cv = StratifiedKFold(
        n_splits=INNER_CV_FOLDS,
        shuffle=True,
        random_state=fold_seed,
    )

    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="accuracy",
        cv=inner_cv,
        n_jobs=N_JOBS,
        refit=True,
    )

    grid.fit(X_train, y_train)

    return grid.best_estimator_, grid.best_params_, grid.best_score_


# Experiment


def run_hybrid_ablation_experiment():
    print("\n=== HaGRID hybrid ablation experiment ===")

    L, y = load_hagrid_landmarks()
    feature_sets, dimensions = build_ablation_features(L)

    dimension_rows = [
        {
            "ablation_variant": name,
            "dimension": dim,
        }
        for name, dim in dimensions.items()
    ]

    dimensions_df = pd.DataFrame(dimension_rows)
    dimensions_path = OUTPUT_DIR / "hagrid_hybrid_ablation_dimensions.csv"
    dimensions_df.to_csv(dimensions_path, index=False)
    print(f"Saved dimensions to: {dimensions_path}")

    outer_cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_SEED,
    )

    rows = []

    split_iter = list(outer_cv.split(np.zeros(len(y)), y))
    total_jobs = len(split_iter) * len(ABLATION_ORDER) * len(CLASSIFIER_ORDER)
    job_count = 0

    for split_id, (train_idx, test_idx) in enumerate(split_iter):
        repeat = split_id // N_SPLITS
        fold = split_id % N_SPLITS
        fold_seed = RANDOM_SEED + 1000 * repeat + fold

        y_train = y[train_idx]
        y_test = y[test_idx]

        print(f"\nOuter repeat {repeat + 1}/{N_REPEATS}, fold {fold + 1}/{N_SPLITS}")

        for ablation_variant in ABLATION_ORDER:
            X = feature_sets[ablation_variant]
            X_train = X[train_idx]
            X_test = X[test_idx]

            for classifier_name in CLASSIFIER_ORDER:
                job_count += 1
                print(f"  [{job_count}/{total_jobs}] {classifier_name} | {ablation_variant}")

                model, best_params, inner_cv_score = fit_tuned_model(
                    classifier_name=classifier_name,
                    X_train=X_train,
                    y_train=y_train,
                    fold_seed=fold_seed,
                )

                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred) * 100.0

                rows.append({
                    "dataset": "HaGRID",
                    "classifier": classifier_name,
                    "ablation_variant": ablation_variant,
                    "repeat": repeat,
                    "fold": fold,
                    "split_id": split_id,
                    "accuracy": acc,
                    "dimension": dimensions[ablation_variant],
                    "inner_cv_accuracy": inner_cv_score * 100.0,
                    "best_params": str(best_params),
                })

    results = pd.DataFrame(rows)

    raw_path = OUTPUT_DIR / "hagrid_hybrid_ablation_results.csv"
    results.to_csv(raw_path, index=False)
    print(f"\nSaved raw results to: {raw_path}")

    summary = (
        results
        .groupby(["dataset", "classifier", "ablation_variant"], as_index=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            min_accuracy=("accuracy", "min"),
            max_accuracy=("accuracy", "max"),
            dimension=("dimension", "first"),
            mean_inner_cv_accuracy=("inner_cv_accuracy", "mean"),
        )
    )

    full_lookup = (
        summary[summary["ablation_variant"] == "Hybrid"]
        .set_index("classifier")["mean_accuracy"]
        .to_dict()
    )

    summary["full_hybrid_accuracy"] = summary["classifier"].map(full_lookup)
    summary["drop_from_full_hybrid"] = (
        summary["full_hybrid_accuracy"] - summary["mean_accuracy"]
    )

    summary["ablation_variant"] = pd.Categorical(
        summary["ablation_variant"],
        categories=ABLATION_ORDER,
        ordered=True,
    )

    summary["classifier"] = pd.Categorical(
        summary["classifier"],
        categories=CLASSIFIER_ORDER,
        ordered=True,
    )

    summary = summary.sort_values(["classifier", "ablation_variant"]).reset_index(drop=True)

    summary_path = OUTPUT_DIR / "hagrid_hybrid_ablation_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary results to: {summary_path}")

    print("\nSummary:")
    print(
        summary[
            [
                "classifier",
                "ablation_variant",
                "dimension",
                "mean_accuracy",
                "std_accuracy",
                "drop_from_full_hybrid",
            ]
        ]
    )

    return results, summary


# Main

if __name__ == "__main__":
    run_hybrid_ablation_experiment()

    print("\nDone. Results saved in:")
    print(OUTPUT_DIR)
