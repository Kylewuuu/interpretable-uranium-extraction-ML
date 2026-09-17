# -*- coding: utf-8 -*-
"""
Benchmark non-XGBoost machine-learning models using the final mixed feature set.

This script:
1. Loads raw input feature files.
2. Loads saved engineered feature names from model_para.
3. Reconstructs the final mixed feature set:
   IF-Property + IF-Fingerprint + IF-Experiment.
4. Benchmarks representative non-XGBoost models.
5. Saves scatter plots and metric tables for SI.

Note:
XGBoost is not included in the benchmark figure here because its performance
is already shown in the main text.
"""

# ============================================================
# Part 0. Imports
# ============================================================

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning


warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================
# Part 1. Basic settings
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Bayesian optimization trials
# 快速测试可先设为 20 或 30；正式结果建议 50 或 100
N_TRIALS = 50

# Input data directory
input_path = "/home/kylewu/hotpot/examples/U_logD/data/input/"

# Saved feature-name directory
save_dir = "/home/kylewu/hotpot/examples/U_logD/model_para"

# Output directory
out_dir = "/home/kylewu/hotpot/examples/U_logD/benchmark_non_xgb_mix"
os.makedirs(out_dir, exist_ok=True)


# ============================================================
# Part 2. Helper functions
# ============================================================

def preprocess_X(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values and standardize feature matrix.

    This follows the preprocessing style used in the original workflow.
    """
    X_filled = SimpleImputer(strategy="mean").fit_transform(df)
    X_scaled = StandardScaler(with_mean=False).fit_transform(X_filled)
    return pd.DataFrame(X_scaled, columns=df.columns)


def preprocess_y(y_array: np.ndarray) -> pd.DataFrame:
    """
    Impute missing values for target variable.
    """
    y_array = y_array.reshape(-1, 1)
    y_filled = SimpleImputer(strategy="mean").fit_transform(y_array).ravel()
    return pd.DataFrame(y_filled, columns=["logD"])


def rmse(y_true, y_pred) -> float:
    return np.sqrt(mean_squared_error(y_true, y_pred))


def check_missing_features(feature_names, df_columns, label):
    """
    Check whether all saved feature names can be found in the reconstructed feature table.
    """
    missing = [f for f in feature_names if f not in df_columns]
    if len(missing) > 0:
        raise ValueError(
            f"\nMissing features in {label}: {missing[:20]}\n"
            f"Total missing: {len(missing)}\n"
            f"Please check whether the raw input files and saved feature names match."
        )


# ============================================================
# Part 3. Load raw data and reconstruct preprocessed feature tables
# ============================================================

# 1. IF-Experiment: experimental condition features
df_lit = pd.read_excel(f"{input_path}sol_envs.xlsx", engine="openpyxl")

# Target
df_y = pd.read_excel(f"{input_path}y.xlsx", engine="openpyxl")
y_df = preprocess_y(df_y["logD"].values)

X_lit_df = preprocess_X(df_lit)


# 2. IF-Property: molecular physicochemical property features
df_ligand_des = pd.read_excel(f"{input_path}ligand_des.xlsx", engine="openpyxl")
df_sol_des = pd.read_excel(f"{input_path}sol_des.xlsx", engine="openpyxl")

X_des_df = preprocess_X(
    pd.concat([df_ligand_des, df_sol_des], axis=1)
)


# 3. IF-Fingerprint: Morgan fingerprint features
df_ligand_mor = pd.read_excel(f"{input_path}ligand_mor.xlsx", engine="openpyxl")
df_sol_mor = pd.read_excel(f"{input_path}sol_mor.xlsx", engine="openpyxl")

X_mor_df = preprocess_X(
    pd.concat([df_ligand_mor, df_sol_mor], axis=1)
)


# ============================================================
# Part 4. Load saved engineered feature names
# ============================================================

# These feature-name files should have been saved after feature engineering
engineered_feature_names_lit = joblib.load(
    os.path.join(save_dir, "engineered_feature_names_lit.pkl")
)

engineered_feature_names_des = joblib.load(
    os.path.join(save_dir, "engineered_feature_names_des.pkl")
)

engineered_feature_names_mor = joblib.load(
    os.path.join(save_dir, "engineered_feature_names_mor.pkl")
)

# Optional: load saved mixed feature-name order if available
mix_feature_path = os.path.join(save_dir, "engineered_feature_names_mix.pkl")
if os.path.exists(mix_feature_path):
    engineered_feature_names_mix_saved = joblib.load(mix_feature_path)
else:
    engineered_feature_names_mix_saved = None


# Check feature availability
check_missing_features(engineered_feature_names_lit, X_lit_df.columns, "IF-Experiment")
check_missing_features(engineered_feature_names_des, X_des_df.columns, "IF-Property")
check_missing_features(engineered_feature_names_mor, X_mor_df.columns, "IF-Fingerprint")


# ============================================================
# Part 5. Construct final mixed feature set
# ============================================================

# Keep the same order as your original mixed model:
# IF-Property + IF-Fingerprint + IF-Experiment
X_des_selected = X_des_df[engineered_feature_names_des]
X_mor_selected = X_mor_df[engineered_feature_names_mor]
X_lit_selected = X_lit_df[engineered_feature_names_lit]

X_mix_df = pd.concat(
    [X_des_selected, X_mor_selected, X_lit_selected],
    axis=1
)

engineered_feature_names_mix = (
    list(engineered_feature_names_des)
    + list(engineered_feature_names_mor)
    + list(engineered_feature_names_lit)
)

# Check saved mixed feature names if available
if engineered_feature_names_mix_saved is not None:
    if list(engineered_feature_names_mix_saved) != engineered_feature_names_mix:
        print(
            "\nWarning: reconstructed mixed feature order is not identical to "
            "engineered_feature_names_mix.pkl."
        )
        print("The script will continue using the reconstructed order:")
        print("IF-Property + IF-Fingerprint + IF-Experiment\n")

X_mix = X_mix_df.values
y_mix = y_df["logD"].values.ravel()

X_train_mix, X_test_mix, y_train_mix, y_test_mix = train_test_split(
    X_mix,
    y_mix,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

kf_mix = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

print("\n===== Final mixed feature set reconstructed =====")
print(f"X_mix shape: {X_mix.shape}")
print(f"Number of IF-Property features: {len(engineered_feature_names_des)}")
print(f"Number of IF-Fingerprint features: {len(engineered_feature_names_mor)}")
print(f"Number of IF-Experiment features: {len(engineered_feature_names_lit)}")
print(f"Total number of mixed features: {len(engineered_feature_names_mix)}")
print(f"Training samples: {X_train_mix.shape[0]}")
print(f"Test samples: {X_test_mix.shape[0]}")


# Save reconstructed mixed feature matrix
X_mix_out = X_mix_df.copy()
X_mix_out["logD"] = y_mix

X_mix_out_path = os.path.join(out_dir, "reconstructed_mixed_feature_matrix.csv")
X_mix_out.to_csv(X_mix_out_path, index=False)

print(f"\nSaved reconstructed mixed feature matrix to:\n{X_mix_out_path}")


# ============================================================
# Part 6. Define non-XGBoost benchmark models
# ============================================================

def build_model(model_name: str, trial: optuna.trial.Trial):
    """
    Build benchmark model using Optuna trial.
    XGBoost is intentionally excluded.
    """

    if model_name == "Ridge":
        return Ridge(
            alpha=trial.suggest_float("alpha", 1e-5, 1e3, log=True)
        )

    elif model_name == "ElasticNet":
        return ElasticNet(
            alpha=trial.suggest_float("alpha", 1e-5, 1e2, log=True),
            l1_ratio=trial.suggest_float("l1_ratio", 0.05, 0.95),
            max_iter=20000,
            random_state=RANDOM_STATE
        )

    elif model_name == "SVR":
        return SVR(
            kernel="rbf",
            C=trial.suggest_float("C", 1e-2, 1e3, log=True),
            epsilon=trial.suggest_float("epsilon", 1e-3, 1.0, log=True),
            gamma=trial.suggest_float("gamma", 1e-5, 1e0, log=True)
        )

    elif model_name == "KNN":
        return KNeighborsRegressor(
            n_neighbors=trial.suggest_int("n_neighbors", 2, 30),
            weights=trial.suggest_categorical("weights", ["uniform", "distance"]),
            p=trial.suggest_int("p", 1, 2)
        )

    elif model_name == "RandomForest":
        return RandomForestRegressor(
            n_estimators=trial.suggest_int("n_estimators", 100, 800),
            max_depth=trial.suggest_int("max_depth", 3, 40),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            max_features=trial.suggest_categorical(
                "max_features", ["sqrt", "log2", 0.5, 0.8, 1.0]
            ),
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    elif model_name == "ExtraTrees":
        return ExtraTreesRegressor(
            n_estimators=trial.suggest_int("n_estimators", 100, 800),
            max_depth=trial.suggest_int("max_depth", 3, 40),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            max_features=trial.suggest_categorical(
                "max_features", ["sqrt", "log2", 0.5, 0.8, 1.0]
            ),
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    elif model_name == "GradientBoosting":
        return GradientBoostingRegressor(
            n_estimators=trial.suggest_int("n_estimators", 50, 600),
            learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            max_depth=trial.suggest_int("max_depth", 2, 8),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            random_state=RANDOM_STATE
        )

    elif model_name == "AdaBoost":
        return AdaBoostRegressor(
            n_estimators=trial.suggest_int("n_estimators", 50, 600),
            learning_rate=trial.suggest_float("learning_rate", 1e-3, 1.0, log=True),
            loss=trial.suggest_categorical("loss", ["linear", "square", "exponential"]),
            random_state=RANDOM_STATE
        )

    elif model_name == "MLP":
        hidden_layer_sizes = trial.suggest_categorical(
            "hidden_layer_sizes",
            [
                (64,),
                (128,),
                (256,),
                (128, 64),
                (256, 128),
                (256, 128, 64)
            ]
        )

        return MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=trial.suggest_categorical("activation", ["relu", "tanh"]),
            alpha=trial.suggest_float("alpha", 1e-6, 1e-1, log=True),
            learning_rate_init=trial.suggest_float("learning_rate_init", 1e-5, 1e-2, log=True),
            max_iter=3000,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=RANDOM_STATE
        )

    else:
        raise ValueError(f"Unknown model name: {model_name}")


def make_objective(model_name: str):
    """
    Objective function for Optuna optimization.
    The objective is five-fold CV MAE on the training set.
    """

    def objective(trial):
        model = build_model(model_name, trial)

        scores = cross_val_score(
            model,
            X_train_mix,
            y_train_mix,
            cv=kf_mix,
            scoring="neg_mean_absolute_error",
            error_score="raise",
            n_jobs=1
        )

        return -np.mean(scores)

    return objective


# ============================================================
# Part 7. Run non-XGBoost benchmark
# ============================================================

model_names = [
    "Ridge",
    "ElasticNet",
    "SVR",
    "KNN",
    "RandomForest",
    "ExtraTrees",
    "GradientBoosting",
    "AdaBoost",
    "MLP"
]

benchmark_results = []
predictions_dict = {}

for model_name in model_names:
    print(f"\n===== Optimizing {model_name} =====")

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    study.optimize(make_objective(model_name), n_trials=N_TRIALS)

    print(f"{model_name} best CV MAE: {study.best_value:.4f}")
    print(f"{model_name} best parameters: {study.best_params}")

    best_model = build_model(model_name, study.best_trial)
    best_model.fit(X_train_mix, y_train_mix)

    train_pred = best_model.predict(X_train_mix)
    test_pred = best_model.predict(X_test_mix)

    train_mae = mean_absolute_error(y_train_mix, train_pred)
    train_rmse = rmse(y_train_mix, train_pred)
    train_r2 = r2_score(y_train_mix, train_pred)

    test_mae = mean_absolute_error(y_test_mix, test_pred)
    test_rmse = rmse(y_test_mix, test_pred)
    test_r2 = r2_score(y_test_mix, test_pred)

    benchmark_results.append({
        "Model": model_name,
        "Feature set": "IF-Experiment + IF-Property + IF-Fingerprint",
        "CV_MAE": study.best_value,
        "Train_R2": train_r2,
        "Train_MAE": train_mae,
        "Train_RMSE": train_rmse,
        "Test_R2": test_r2,
        "Test_MAE": test_mae,
        "Test_RMSE": test_rmse,
        "Best_params": str(study.best_params)
    })

    predictions_dict[model_name] = {
        "y_train": y_train_mix,
        "train_pred": train_pred,
        "y_test": y_test_mix,
        "test_pred": test_pred,
        "test_r2": test_r2,
        "test_mae": test_mae,
        "test_rmse": test_rmse
    }


# ============================================================
# Part 8. Save benchmark metrics
# ============================================================

benchmark_df = pd.DataFrame(benchmark_results)

benchmark_csv = os.path.join(
    out_dir,
    "Table_S4_non_xgb_benchmark_mix_metrics.csv"
)

benchmark_df.to_csv(benchmark_csv, index=False)

print("\n===== Benchmark summary =====")
print(
    benchmark_df[
        [
            "Model",
            "CV_MAE",
            "Train_R2",
            "Train_MAE",
            "Train_RMSE",
            "Test_R2",
            "Test_MAE",
            "Test_RMSE"
        ]
    ]
)

print(f"\nSaved benchmark metrics to:\n{benchmark_csv}")


# ============================================================
# Part 9. Save detailed prediction results
# ============================================================

prediction_rows = []

for model_name in model_names:
    result = predictions_dict[model_name]

    for y_true, y_pred in zip(result["y_train"], result["train_pred"]):
        prediction_rows.append({
            "Model": model_name,
            "Set": "Train",
            "Experimental_logD": y_true,
            "Predicted_logD": y_pred
        })

    for y_true, y_pred in zip(result["y_test"], result["test_pred"]):
        prediction_rows.append({
            "Model": model_name,
            "Set": "Test",
            "Experimental_logD": y_true,
            "Predicted_logD": y_pred
        })

pred_df = pd.DataFrame(prediction_rows)

pred_csv = os.path.join(
    out_dir,
    "non_xgb_benchmark_mix_predictions.csv"
)

pred_df.to_csv(pred_csv, index=False)

print(f"Saved detailed predictions to:\n{pred_csv}")


# ============================================================
# Part 10. Save SI-ready metric table
# ============================================================

si_table = benchmark_df[
    [
        "Model",
        "CV_MAE",
        "Test_R2",
        "Test_MAE",
        "Test_RMSE"
    ]
].copy()

si_table = si_table.rename(columns={
    "CV_MAE": "CV MAE",
    "Test_R2": "Test R²",
    "Test_MAE": "Test MAE",
    "Test_RMSE": "Test RMSE"
})

# Sort by Test R² from high to low
si_table = si_table.sort_values(by="Test R²", ascending=False)

si_table_csv = os.path.join(
    out_dir,
    "Table_S4_SI_ready_non_xgb_metrics.csv"
)

si_table.to_csv(si_table_csv, index=False)

print(f"\nSaved SI-ready metric table to:\n{si_table_csv}")
print("\nSI-ready metrics:")
print(si_table)


# ============================================================
# Part 11. Plot scatter plots for SI
# ============================================================

plot_models = model_names

all_true = []
all_pred = []

for model_name in plot_models:
    result = predictions_dict[model_name]

    all_true.extend(result["y_train"])
    all_true.extend(result["y_test"])
    all_pred.extend(result["train_pred"])
    all_pred.extend(result["test_pred"])

min_lim = min(min(all_true), min(all_pred)) - 0.2
max_lim = max(max(all_true), max(all_pred)) + 0.2

n_models = len(plot_models)
n_cols = 3
n_rows = int(np.ceil(n_models / n_cols))

fig, axes = plt.subplots(
    n_rows,
    n_cols,
    figsize=(4.0 * n_cols, 3.8 * n_rows)
)

axes = np.array(axes).reshape(-1)

for ax, model_name in zip(axes, plot_models):
    result = predictions_dict[model_name]

    ax.scatter(
        result["y_train"],
        result["train_pred"],
        s=18,
        alpha=0.55,
        label="Training set"
    )

    ax.scatter(
        result["y_test"],
        result["test_pred"],
        s=28,
        alpha=0.85,
        marker="^",
        label="Test set"
    )

    ax.plot(
        [min_lim, max_lim],
        [min_lim, max_lim],
        linestyle="--",
        linewidth=1.2
    )

    ax.set_xlim(min_lim, max_lim)
    ax.set_ylim(min_lim, max_lim)
    ax.set_aspect("equal", adjustable="box")

    ax.set_title(model_name, fontsize=11, fontweight="bold")
    ax.set_xlabel("Experimental logD(U)", fontsize=10)
    ax.set_ylabel("Predicted logD(U)", fontsize=10)

    metric_text = (
        f"Test R² = {result['test_r2']:.3f}\n"
        f"MAE = {result['test_mae']:.3f}\n"
        f"RMSE = {result['test_rmse']:.3f}"
    )

    ax.text(
        0.04,
        0.96,
        metric_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9
    )

    ax.tick_params(labelsize=9)

# Hide unused axes
for ax in axes[len(plot_models):]:
    ax.axis("off")

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=2,
    frameon=False,
    fontsize=10,
    bbox_to_anchor=(0.5, 1.02)
)

fig.tight_layout(rect=[0, 0, 1, 0.97])

fig_png = os.path.join(
    out_dir,
    "Fig_S1_non_xgb_benchmark_mix_scatter.png"
)

fig_pdf = os.path.join(
    out_dir,
    "Fig_S1_non_xgb_benchmark_mix_scatter.pdf"
)

fig_tiff = os.path.join(
    out_dir,
    "Fig_S1_non_xgb_benchmark_mix_scatter.tiff"
)

fig.savefig(fig_png, dpi=600, bbox_inches="tight")
fig.savefig(fig_pdf, bbox_inches="tight")
fig.savefig(fig_tiff, dpi=600, bbox_inches="tight")

plt.show()

print("\nSaved SI benchmark figure to:")
print(fig_png)
print(fig_pdf)
print(fig_tiff)