import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

PICKLE_FILE = "data.pickle"
OUTPUT_DIR = "eda_outputs"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


with open(PICKLE_FILE, "rb") as f:
    dataset = pickle.load(f)

X = np.array(dataset["data"])
y = np.array(dataset["labels"])

print("Dataset loaded successfully.")
print(f"Number of observations: {X.shape[0]}")
print(f"Number of predictors: {X.shape[1]}")
print(f"Number of classes: {len(np.unique(y))}")



feature_names = []

for i in range(21):
    feature_names.append(f"x{i}")
    feature_names.append(f"y{i}")

df = pd.DataFrame(X, columns=feature_names)
df["label"] = y

print("\nFirst few rows:")
print(df.head())


print("\nDataset shape:")
print(df.shape)

print("\nClass labels:")
print(sorted(df["label"].unique()))

print("\nMissing values per column:")
print(df.isnull().sum())

summary_stats = df.drop(columns=["label"]).describe().T
summary_stats.to_csv(os.path.join(OUTPUT_DIR, "summary_statistics.csv"))

print("\nSummary statistics saved to eda_outputs/summary_statistics.csv")


class_counts = df["label"].value_counts().sort_index()
class_percentages = df["label"].value_counts(normalize=True).sort_index() * 100

class_balance = pd.DataFrame({
    "count": class_counts,
    "percentage": class_percentages.round(2)
})

class_balance.to_csv(os.path.join(OUTPUT_DIR, "class_balance.csv"))

print("\nClass balance:")
print(class_balance)


plt.figure(figsize=(10, 5))
class_counts.plot(kind="bar")
plt.xlabel("Gesture Class")
plt.ylabel("Number of Images")
plt.title("Number of Images per Gesture Class")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "class_counts.png"), dpi=300)
plt.close()

print("Saved class count plot.")



x_columns = [col for col in feature_names if col.startswith("x")]
y_columns = [col for col in feature_names if col.startswith("y")]

x_values = df[x_columns].values.flatten()
y_values = df[y_columns].values.flatten()

plt.figure(figsize=(8, 5))
plt.hist(x_values, bins=30, alpha=0.7, label="x-coordinates")
plt.hist(y_values, bins=30, alpha=0.7, label="y-coordinates")
plt.xlabel("Normalized Coordinate Value")
plt.ylabel("Frequency")
plt.title("Distribution of Normalized Landmark Coordinates")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "coordinate_distribution.png"), dpi=300)
plt.close()

print("Saved coordinate distribution plot.")



def plot_hand_landmarks(row, title, save_path):
    x_coords = []
    y_coords = []

    for i in range(21):
        x_coords.append(row[f"x{i}"])
        y_coords.append(row[f"y{i}"])

    plt.figure(figsize=(5, 5))
    plt.scatter(x_coords, y_coords)

    for i in range(21):
        plt.text(x_coords[i], y_coords[i], str(i), fontsize=8)

    plt.gca().invert_yaxis()
    plt.xlabel("Normalized x-coordinate")
    plt.ylabel("Normalized y-coordinate")
    plt.title(title)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


for label in sorted(df["label"].unique()):
    example_row = df[df["label"] == label].iloc[0]

    save_path = os.path.join(
        OUTPUT_DIR,
        f"example_landmarks_class_{label}.png"
    )

    plot_hand_landmarks(
        example_row,
        f"Example Hand Landmarks for Class {label}",
        save_path
    )

print("Saved one landmark plot per class.")


for label in sorted(df["label"].unique()):
    class_df = df[df["label"] == label]
    mean_row = class_df.drop(columns=["label"]).mean()

    save_path = os.path.join(
        OUTPUT_DIR,
        f"mean_landmarks_class_{label}.png"
    )

    plot_hand_landmarks(
        mean_row,
        f"Mean Hand Landmark Shape for Class {label}",
        save_path
    )

print("Saved mean landmark plot per class.")



plt.figure(figsize=(7, 7))

for label in sorted(df["label"].unique()):
    class_df = df[df["label"] == label]
    mean_row = class_df.drop(columns=["label"]).mean()

    x_coords = []
    y_coords = []

    for i in range(21):
        x_coords.append(mean_row[f"x{i}"])
        y_coords.append(mean_row[f"y{i}"])

    plt.plot(x_coords, y_coords, marker="o", label=f"Class {label}")

plt.gca().invert_yaxis()
plt.xlabel("Mean normalized x-coordinate")
plt.ylabel("Mean normalized y-coordinate")
plt.title("Mean Landmark Shapes by Gesture Class")
plt.legend()
plt.axis("equal")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "mean_landmarks_by_class.png"), dpi=300)
plt.close()

print("Saved comparison of mean landmark shapes.")



selected_label = "0"

selected_df = df[df["label"] == selected_label]

if len(selected_df) == 0:
    print(f"No observations found for class {selected_label}.")
else:
    mean_row = selected_df.drop(columns=["label"]).mean()

    x_coords = []
    y_coords = []

    for i in range(21):
        x_coords.append(mean_row[f"x{i}"])
        y_coords.append(mean_row[f"y{i}"])

    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
        (0, 5), (5, 6), (6, 7), (7, 8),          # index finger
        (0, 9), (9, 10), (10, 11), (11, 12),     # middle finger
        (0, 13), (13, 14), (14, 15), (15, 16),   # ring finger
        (0, 17), (17, 18), (18, 19), (19, 20)    # pinky
    ]

    save_path = os.path.join(
        OUTPUT_DIR,
        f"selected_mean_landmarks_class_{selected_label}.png"
    )

    plt.figure(figsize=(5, 5))

    plt.scatter(x_coords, y_coords)

    for start, end in connections:
        plt.plot(
            [x_coords[start], x_coords[end]],
            [y_coords[start], y_coords[end]]
        )

    for i in range(21):
        plt.text(x_coords[i], y_coords[i], str(i), fontsize=8)

    plt.gca().invert_yaxis()
    plt.xlabel("Mean normalized x-coordinate")
    plt.ylabel("Mean normalized y-coordinate")
    plt.title(f"Connected Mean Hand Landmark Shape for Class {selected_label}")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

X_features = df.drop(columns=["label"]).values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

pca_summary = pd.DataFrame({
    "principal_component": [f"PC{i+1}" for i in range(len(explained_variance))],
    "proportion_variance_explained": explained_variance,
    "cumulative_variance_explained": cumulative_variance
})

pca_summary.to_csv(os.path.join(OUTPUT_DIR, "pca_summary.csv"), index=False)

print("\nPCA summary saved to eda_outputs/pca_summary.csv")



plt.figure(figsize=(8, 5))
plt.plot(
    range(1, len(explained_variance) + 1),
    explained_variance,
    marker="o"
)
plt.xlabel("Principal Component")
plt.ylabel("Proportion of Variance Explained")
plt.title("PCA Scree Plot")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "pca_scree_plot.png"), dpi=300)
plt.close()

print("Saved PCA scree plot.")



plt.figure(figsize=(8, 5))
plt.plot(
    range(1, len(cumulative_variance) + 1),
    cumulative_variance,
    marker="o"
)
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Proportion of Variance Explained")
plt.title("Cumulative Variance Explained by PCA")
plt.axhline(y=0.80, linestyle="--", label="80%")
plt.axhline(y=0.90, linestyle="--", label="90%")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "pca_cumulative_variance.png"), dpi=300)
plt.close()

print("Saved PCA cumulative variance plot.")



pca_df = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "label": y
})

plt.figure(figsize=(8, 6))

for label in sorted(pca_df["label"].unique()):
    subset = pca_df[pca_df["label"] == label]
    plt.scatter(
        subset["PC1"],
        subset["PC2"],
        label=f"Class {label}",
        alpha=0.7
    )

plt.xlabel(f"PC1 ({explained_variance[0] * 100:.1f}% variance)")
plt.ylabel(f"PC2 ({explained_variance[1] * 100:.1f}% variance)")
plt.title("PCA Projection of Hand Landmark Data")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "pca_pc1_pc2_by_class.png"), dpi=300)
plt.close()

print("Saved PCA PC1 vs PC2 plot.")


corr_matrix = df.drop(columns=["label"]).corr()

plt.figure(figsize=(12, 10))
plt.imshow(corr_matrix, aspect="auto")
plt.colorbar(label="Correlation")
plt.xticks(range(len(feature_names)), feature_names, rotation=90, fontsize=6)
plt.yticks(range(len(feature_names)), feature_names, fontsize=6)
plt.title("Correlation Heatmap of Landmark Features")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_correlation_heatmap.png"), dpi=300)
plt.close()

print("Saved feature correlation heatmap.")


with open(os.path.join(OUTPUT_DIR, "eda_summary.txt"), "w") as f:
    f.write("EDA Summary\n")
    f.write("====================\n\n")
    f.write(f"Number of observations: {X.shape[0]}\n")
    f.write(f"Number of predictors: {X.shape[1]}\n")
    f.write(f"Number of classes: {len(np.unique(y))}\n\n")

    f.write("Class balance:\n")
    f.write(class_balance.to_string())
    f.write("\n\n")

    f.write("PCA results:\n")
    f.write(f"PC1 explains {explained_variance[0] * 100:.2f}% of the variance.\n")
    f.write(f"PC2 explains {explained_variance[1] * 100:.2f}% of the variance.\n")
    f.write(f"PC1 and PC2 together explain {(explained_variance[0] + explained_variance[1]) * 100:.2f}% of the variance.\n")

    num_components_80 = np.argmax(cumulative_variance >= 0.80) + 1
    num_components_90 = np.argmax(cumulative_variance >= 0.90) + 1

    f.write(f"Number of PCs needed for 80% variance: {num_components_80}\n")
    f.write(f"Number of PCs needed for 90% variance: {num_components_90}\n")

print("Saved written EDA summary.")


print("\nEDA complete.")
print(f"All outputs saved in: {OUTPUT_DIR}")
