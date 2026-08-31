from pathlib import Path
import re
import pickle
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "final_results_eda"
OUTPUT_DIR.mkdir(exist_ok=True)

SELF_LANDMARK_FILE = PROJECT_ROOT / "Representations" / "raw_xy.pickle"
SELF_LABEL_FILE = None



RANDOM_SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
PERTURBATION_REPEATS = 10

PERTURBATION_LEVELS = {
    0: {"translation": 0.00, "scale_low": 1.00, "scale_high": 1.00, "noise": 0.000},
    1: {"translation": 0.10, "scale_low": 0.80, "scale_high": 1.25, "noise": 0.001},
    2: {"translation": 0.25, "scale_low": 0.60, "scale_high": 1.60, "noise": 0.002},
    3: {"translation": 0.45, "scale_low": 0.40, "scale_high": 2.00, "noise": 0.003},
    4: {"translation": 0.70, "scale_low": 0.30, "scale_high": 2.50, "noise": 0.004},
}

REPRESENTATIONS = [
    "XY",
    "Translated",
    "Scaled",
    "Norm. distance",
    "Angles",
    "Hybrid",
]

warnings.filterwarnings("ignore")



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
    if SELF_LABEL_FILE is not None:
        return load_label_file(SELF_LABEL_FILE, expected_n)

    source_path = Path(source_path)

    search_dirs = [
        source_path.parent,
        PROJECT_ROOT / "Representations",
        PROJECT_ROOT / "representations",
        PROJECT_ROOT / "data",
    ]

    label_names = [
        "labels", "label", "y", "targets", "target", "classes", "class", "gestures", "gesture"
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
        f"No matching label file was found. Set SELF_LABEL_FILE at the top."
    )


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


def load_self_landmarks():
    path = Path(SELF_LANDMARK_FILE)

    if not path.exists():
        raise FileNotFoundError(f"SELF_LANDMARK_FILE does not exist: {path}")

    obj = read_any(path)
    L, y = object_to_landmarks_and_labels(obj, source_path=path)

    print(f"Loaded landmarks from: {path}")
    print(f"Landmarks shape: {L.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Classes: {len(np.unique(y))}")

    return L, y



EPS = 1e-8

PAIRWISE_PAIRS = [(i, j) for i in range(21) for j in range(i + 1, 21)]

ANGLE_TRIPLES = [
    (0, 1, 2), (1, 2, 3), (2, 3, 4),
    (0, 5, 6), (5, 6, 7), (6, 7, 8),
    (0, 9, 10), (9, 10, 11), (10, 11, 12),
    (0, 13, 14), (13, 14, 15), (14, 15, 16),
    (0, 17, 18), (17, 18, 19), (18, 19, 20),
]


def phi_xy(L):
    return L.reshape(L.shape[0], -1)


def phi_translated(L):
    mins = L.min(axis=1, keepdims=True)
    return (L - mins).reshape(L.shape[0], -1)


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


def phi_hybrid(L):
    return np.concatenate(
        [
            phi_scaled(L),
            phi_norm_distance(L),
            phi_angles(L),
        ],
        axis=1,
    )


FEATURE_FUNCTIONS = {
    "XY": phi_xy,
    "Translated": phi_translated,
    "Scaled": phi_scaled,
    "Norm. distance": phi_norm_distance,
    "Angles": phi_angles,
    "Hybrid": phi_hybrid,
}



def fit_logistic_model(X_train, y_train):
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

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )

    grid.fit(X_train, y_train)

    return grid.best_estimator_, grid.best_params_, grid.best_score_



def perturb_landmarks(L, config, rng):
    translation_amount = config["translation"]
    scale_low = config["scale_low"]
    scale_high = config["scale_high"]
    noise_amount = config["noise"]

    n = L.shape[0]

    translations = rng.uniform(
        low=-translation_amount,
        high=translation_amount,
        size=(n, 1, 2),
    )

    scales = rng.uniform(
        low=scale_low,
        high=scale_high,
        size=(n, 1, 1),
    )

    noise = rng.normal(
        loc=0.0,
        scale=noise_amount,
        size=L.shape,
    )

    return scales * L + translations + noise


def run_experiment():
    print("\n=== Synthetic perturbation robustness experiment ===")

    L, y = load_self_landmarks()

    indices = np.arange(len(y))

    train_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_SEED,
    )

    L_train = L[train_idx]
    L_test = L[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    trained_models = {}

    for rep in REPRESENTATIONS:
        print(f"\nTraining clean model for {rep}")

        X_train = FEATURE_FUNCTIONS[rep](L_train)
        model, best_params, best_cv = fit_logistic_model(X_train, y_train)

        trained_models[rep] = model

        print(f"  best CV: {best_cv * 100:.2f}%")
        print(f"  best params: {best_params}")

    rows = []

    for level, config in PERTURBATION_LEVELS.items():
        print(f"\nPerturbation level {level}: {config}")

        for repeat in range(PERTURBATION_REPEATS):
            rng = np.random.default_rng(RANDOM_SEED + 1000 * level + repeat)
            L_test_perturbed = perturb_landmarks(L_test, config, rng)

            for rep in REPRESENTATIONS:
                X_test = FEATURE_FUNCTIONS[rep](L_test_perturbed)
                y_pred = trained_models[rep].predict(X_test)
                acc = accuracy_score(y_test, y_pred) * 100.0

                rows.append(
                    {
                        "level": level,
                        "repeat": repeat,
                        "representation": rep,
                        "accuracy": acc,
                        "translation": config["translation"],
                        "scale_low": config["scale_low"],
                        "scale_high": config["scale_high"],
                        "noise": config["noise"],
                    }
                )

    results = pd.DataFrame(rows)

    raw_path = OUTPUT_DIR / "synthetic_perturbation_robustness_results.csv"
    results.to_csv(raw_path, index=False)
    print(f"\nSaved raw results to: {raw_path}")

    summary = (
        results
        .groupby(["level", "representation"], as_index=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
        )
    )

    summary_path = OUTPUT_DIR / "synthetic_perturbation_robustness_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary results to: {summary_path}")

    return results, summary



if __name__ == "__main__":
    run_experiment()

    print("\nDone. Results saved in:")
    print(OUTPUT_DIR)