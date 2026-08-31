
## Datasets

There are scripts in here for a few different datasets.

### Self-collected ASL data

`data_collection_creation/Image_Collection_Script.py` is the script I used to collect images from a webcam.

The main self-collected dataset is expected in:

```text
data/
```

### SignAlphaSet

The SignAlphaSet scripts use:

```text
SignAlphaSet_data/
```

`Data_Creation_SignALphaSet_Script.py` processes the images and currently uses up to 100 images per class.

### HaGRID

`download_hagrid_100_per_class.py` downloads the HaGRID subset from Hugging Face:

```text
GestureDetectionConnoisseurs/hagrid_subsets
```

and saves up to 100 images per class in:

```text
hagrid_100_data/
```

### Kaggle data

There are also scripts set up for images stored in:

```text
kaggle_data/
```

The actual image datasets, pickles, and trained models are not included in the repo because they get pretty large.

## Classification

I used two main classifiers.

### Multinomial Logistic Regression

The logistic regression scripts use `StandardScaler` followed by `LogisticRegression`.

The values of `C` tested are:

```python
C = [0.001, 0.01, 0.1, 1, 10, 100]
```

with 5-fold stratified cross validation.

The model uses `lbfgs` and allows up to 10,000 iterations.

### Random Forest

For the random forest I tested:

```python
n_estimators = [100, 300, 500]
max_features = ["sqrt", "log2"]
max_depth = [None, 10, 20]
min_samples_leaf = [1, 2, 4]
```

also using 5-fold stratified cross validation.

The main train/test scripts use an 80/20 stratified split with:

```python
random_state = 42
```

## Repository Structure

The project is roughly organized like this:

```text
.
├── configuration_scripts/
│   ├── Multinomial_Regression_General_Config_K_Fold_Script.py
│   ├── Random_Forest_General_Config_K_Fold_Script.py
│   ├── SignAlphaSet_*_Config_*.py
│   ├── Hagrid_*_Config_*.py
│   ├── Kaggle_*_Config_*.py
│   └── *_Friedman_Willcox_Accuracry_Config_Script.py
│
├── data_collection_creation/
│   ├── Image_Collection_Script.py
│   ├── Data_Creation_Script.py
│   ├── Data_Creation_SignALphaSet_Script.py
│   ├── Data_Creation_Hagrid_Script.py
│   └── Data_Creation_Kaggle_Script.py
│
├── final_eda_scripts/
│   ├── hagrid_hybrid_ablation_experiment.py
│   ├── hybrid_synthetic_robustness_intensity_experiment_script.py
│   ├── hybrid_synthetic_robustness_type_experiment_script.py
│   ├── final_results_visualisations_script.py
│   ├── heatmap_scripts.py
│   ├── representation_average_rank_script.py
│   ├── statistical_visuals.py
│   └── ...
│
├── final_results_eda/
│   ├── combined_accuracy_results.csv
│   ├── hagrid_repeated_cv_summary.csv
│   ├── hagrid_hybrid_ablation_summary.csv
│   ├── perturbation_type_summary.csv
│   └── ...
│
├── software_demos_scripts/
│   ├── Classifier_Demo_Script.py
│   ├── MLR_Sentence_builder_demo.py
│   └── NN_Sentence_Builder_Demo.py
│
├── download_hagrid_100_per_class.py
├── Eda_Script.py
└── .gitignore
```

There are a bunch of generated folders locally too for models, feature representations, results, and figures. Those are not all meant to be committed.

## Installation

Clone the repo:

```bash
git clone <repository-url>
cd <repository-name>
```

Create a virtual environment if you want one.

I use Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```


Then install the requirements:

```bash
pip install -r requirements.txt
```

Main packages used in the project are:

- NumPy
- pandas
- Matplotlib
- scikit-learn
- SciPy
- OpenCV
- MediaPipe
- joblib
- Hugging Face `datasets`
- Pillow

TensorFlow is only used for the neural network code/demo.

## Data Layout

Most of the data creation scripts expect one folder per class.

For example:

```text
data/
├── A/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
├── B/
│   └── ...
└── ...
```

The folder name is used as the label.

The default folders are:

```text
data/                 -> self-collected dataset
SignAlphaSet_data/    -> SignAlphaSet
hagrid_100_data/      -> HaGRID subset
kaggle_data/          -> Kaggle dataset
```

## Running the Experiments

A lot of these scripts were written with relative paths, so it is usually easiest to run them from the folder they are already in.

### Download HaGRID

```bash
python ../download_hagrid_100_per_class.py
```

This downloads up to 100 images per class.

### Generate the feature representations

From `data_collection_creation/`:

Self-collected data:

```bash
python Data_Creation_Script.py
```

SignAlphaSet:

```bash
python Data_Creation_SignALphaSet_Script.py
```

HaGRID:

```bash
python Data_Creation_Hagrid_Script.py
```

Kaggle:

```bash
python Data_Creation_Kaggle_Script.py
```

These scripts generate files like:

```text
raw_xy.pickle
raw_xyz.pickle
translated_xy.pickle
scaled_xy.pickle
distances_xy.pickle
normalized_distances_xy.pickle
angles_xy.pickle
hybrid.pickle
```

The pickle files are stored like:

```python
{
    "data": feature_vectors,
    "labels": class_labels
}
```

### Train the classifiers

From `configuration_scripts/`:

Self-collected:

```bash
python Multinomial_Regression_General_Config_K_Fold_Script.py
python Random_Forest_General_Config_K_Fold_Script.py
```

SignAlphaSet:

```bash
python SignAlphaSet_Multinomial_Regression_General_Config_K_Fold_Script.py
python SignAlphaSet_Random_Forest_General_Config_K_Fold_Script.py
```

HaGRID:

```bash
python Hagrid_Multinomial_Regression_General_Config_K_Fold_Script.py
python Hagrid_Random_Forest_General_Config_K_Fold_Script.py
```

Kaggle:

```bash
python Kaggle_Multinomial_Regression_General_Config_K_Fold_Script.py
python Kaggle_Random_Forest_General_Config_K_Fold_Script.py
```

These scripts save the model results, classification reports, confusion matrices, hyperparameter results, and trained models.

### Statistical tests

There are also scripts for the repeated cross-validation / Friedman / Wilcoxon comparisons.

Examples:

```bash
python Hagrid_Friedman_Willcox_Accuracry_Config_Script.py
python SignAlphaSet_Friedman_Willcox_Accuracry_Config_Script.py
python Friedman_Willcox_Accuracry_Config_Script.py
```

### Figures

Most of the final plotting scripts are in:

```text
final_eda_scripts/
```

There are scripts in there for the heatmaps, HaGRID plots, robustness plots, representation rankings, repeated CV results, perturbation figures, ablation figures, and a few other paper figures.

## Real-Time Demo

`software_demos_scripts/MLR_Sentence_builder_demo.py` is the webcam demo.

It recognizes:

```text
A-Z, Space, Clear, Enter
```

and builds up a sentence as predictions are made.

It expects these saved files in `models/`:

```text
multinomial_logistic_regression_model.joblib
logistic_regression_scaler.joblib
label_encoder.joblib
```

Run it with:

```bash
python MLR_Sentence_builder_demo.py
```

Press `q` to quit.

One thing to keep in mind is that the demo builds a 42-dimensional translated-coordinate feature vector, so the saved model and scaler need to match that representation.

## Outputs

Depending on the script, the project generates things like:

- `.joblib` trained models
- `.pickle` feature representations
- confusion matrices
- classification reports
- hyperparameter results
- logistic regression coefficients
- random forest feature importances
- cross-validation results
- Friedman / Wilcoxon results
- perturbation results
- ablation results
- PNG, PDF, and SVG figures

Some of the final CSV summaries are kept in:

```text
final_results_eda/
```

## A Few Notes

- Most randomized experiments use `random_state=42`.
- The classification scripts use stratified splits/cross validation.
- MediaPipe uses a minimum detection confidence of `0.3` in the data creation scripts.
- Images where MediaPipe does not find a hand are skipped.
- Only the first detected hand is used.
- Some scripts use relative paths based on the way I had the project set up on Windows, so those may need to be changed depending on where the repo is cloned.

## Dependencies

The current `requirements.txt` is:

```text
numpy
pandas
matplotlib
scikit-learn
scipy
opencv-python
mediapipe
joblib
datasets
Pillow
tensorflow
```

TensorFlow is only needed for the neural network stuff. The main logistic regression / random forest experiments do not need it.

## Citation

This repo is connected to the paper:

*Beyond Landmark Extraction: A Framework for Robust Geometric Feature Construction in Structured Image Classification*

If you use the code for academic work, the citation can be updated here once the paper is published.

```bibtex
@article{landmark_feature_representations,
  title   = {Beyond Landmark Extraction: A Framework for Robust Geometric Feature Construction in Structured Image Classification},
  author  = {Mauree, Saravana and Arya, Sakshi},
  journal = {To appear},
  year    = {2026}
}
```
