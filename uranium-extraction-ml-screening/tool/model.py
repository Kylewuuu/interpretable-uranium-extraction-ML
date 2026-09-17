import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from hotpot.plugins.ml.wf import MachineLearning # 用于特征工程
from hotpot.plugins.ml.wf import MachineLearning_ # 用于获取聚类重要特征
#%%
def preprocess_X(df):
    X_filled = SimpleImputer(strategy='mean').fit_transform(df)
    X_scaled = StandardScaler(with_mean=False).fit_transform(X_filled)
    return pd.DataFrame(X_scaled, columns=df.columns)

def preprocess_y(y_array):
    y_array = y_array.reshape(-1, 1)
    y_filled = SimpleImputer(strategy='mean').fit_transform(y_array).ravel()
    return pd.DataFrame(y_filled, columns=['logD'])

# 设置路径
input_path = '/home/kylewu/hotpot/examples/U_logD/data/input/'

# 1. Literature Data (sol_envs.xlsx)
df_lit = pd.read_excel(f'{input_path}sol_envs.xlsx', engine='openpyxl')
df_y = pd.read_excel(f'{input_path}y.xlsx', engine='openpyxl')
X_lit_df = preprocess_X(df_lit)
y_lit_df = preprocess_y(df_y['logD'].values)

# 2. Descriptor Data (ligand_des.xlsx + sol_des.xlsx)
df_ligand_des = pd.read_excel(f'{input_path}ligand_des.xlsx', engine='openpyxl')
df_sol_des = pd.read_excel(f'{input_path}sol_des.xlsx', engine='openpyxl')
X_des_df = preprocess_X(pd.concat([df_ligand_des, df_sol_des], axis=1))
y_des_df = preprocess_y(df_y['logD'].values)

# 3. Morgan Data (ligand_mor.xlsx + sol_mor.xlsx)
df_ligand_mor = pd.read_excel(f'{input_path}ligand_mor.xlsx', engine='openpyxl')
df_sol_mor = pd.read_excel(f'{input_path}sol_mor.xlsx', engine='openpyxl')
X_mor_df = preprocess_X(pd.concat([df_ligand_mor, df_sol_mor], axis=1))
y_mor_df = preprocess_y(df_y['logD'].values)

#%%
"""Part 2 Feature engineering"""
# Define the XGBoost regression model object and set the parameters
model = XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.6,
    colsample_bytree=1.0,
    reg_alpha=1, reg_lambda=10,
    random_state=42
)
# Define working directories and feature lists
# Add the target variable to the feature data

"""literature data"""
work_dir_lit = '/home/kylewu/hotpot/examples/U_logD/ML_lit'
features_lit = X_lit_df.columns.tolist()  # need to obtain column names from the feature data
data_lit = pd.concat([X_lit_df, y_lit_df], axis=1)
"""literature data"""

"""descriptor data"""
work_dir_des = '/home/kylewu/hotpot/examples/U_logD/ML_des'
features_des = X_des_df.columns.tolist()
data_des = pd.concat([X_des_df, y_des_df], axis=1)
"""descriptor data"""

"""morgan data"""
work_dir_mor = '/home/kylewu/hotpot/examples/U_logD/ML_mor'
features_mor = X_mor_df.columns.tolist()
data_mor = pd.concat([X_mor_df, y_mor_df], axis=1)
"""morgan data"""



# Initialize the MachineLearning object

"""literature data"""
ml_lit = MachineLearning(
    work_dir=work_dir_lit,
    data=data_lit,
    features=features_lit,
    target='logD',
    estimator=model,
    #xscaler=scaler  # If you don't need a pantograph, you can omit this parameter
)

ml_lit.preprocess()

ml_lit.train_test_split()

ml_lit.quickly_feature_selection()

ml_lit.calc_pearson_matrix()

ml_lit.feat_hiera_clustering()
"""literature data"""

#%%
"""descriptor data"""
ml_des = MachineLearning(
    work_dir=work_dir_des,
    data=data_des,
    features=features_des,
    target='logD',
    estimator=model,
    #xscaler=scaler
)

ml_des.preprocess()

ml_des.train_test_split()

ml_des.quickly_feature_selection()

ml_des.calc_pearson_matrix()

# ml_des.pca_dimension_reduction()

ml_des.feat_hiera_clustering()
"""descriptor data"""


"""morgan data"""
ml_mor = MachineLearning(
    work_dir=work_dir_mor,
    data=data_mor,
    features=features_mor,
    target='logD',
    estimator=model,
    #xscaler=scaler
)

ml_mor.preprocess()

ml_mor.train_test_split()

ml_mor.quickly_feature_selection()

ml_mor.calc_pearson_matrix()

# ml_mor.pca_dimension_reduction()

ml_mor.feat_hiera_clustering()
"""morgan data"""

#%%
"""literature data"""
# Get the feature indices after clustering
feature_indices_lit = ml_lit.ntri_feat_indices  # Assume this is the indices of non-trivial features

# Get feature names
features_lit = np.array(ml_lit.features)

# Create a dictionary mapping cluster labels to feature names
clustered_feat_names_lit = ml_lit.clustered_feat_names  # Get clustering results from the code provided

# Print each index corresponding to the feature
for cluster_idx in sorted(clustered_feat_names_lit.keys()):
    print(f"Cluster {cluster_idx}: {clustered_feat_names_lit[cluster_idx]}")

# Iterate through each cluster label and its corresponding feature indices
for cluster_label, indices in ml_lit.ntri_feat_cluster.items():
    feature_info_lit = [(idx, ml_lit.features[idx]) for idx in indices if idx < len(ml_lit.features)]  # Get indices and feature names
    print(f"Cluster {cluster_label}: {feature_info_lit}")

# Initialize a set to store features
final_features_lit = set()

# Extract the last feature of each cluster
for cluster_idx in sorted(clustered_feat_names_lit.keys()):
    features_in_cluster_lit = clustered_feat_names_lit[cluster_idx]
    if isinstance(features_in_cluster_lit, list):
        final_features_lit.add(features_in_cluster_lit[-1])  # Take the last feature
    else:
        final_features_lit.add(features_in_cluster_lit)  # If there's only one feature, add it directly

# Convert the set to a list
final_features_list_lit = list(final_features_lit)

# Select these features from the original data and merge with the target variable
data_1_lit = pd.concat([data_lit[final_features_list_lit], y_lit_df], axis=1)
"""literature data"""


"""descriptor data"""
feature_indices_des = ml_des.ntri_feat_indices

features_des = np.array(ml_des.features)

clustered_feat_names_des = ml_des.clustered_feat_names

for cluster_idx in sorted(clustered_feat_names_des.keys()):
    print(f"Cluster {cluster_idx}: {clustered_feat_names_des[cluster_idx]}")

for cluster_label, indices in ml_des.ntri_feat_cluster.items():
    feature_info_des = [(idx, ml_des.features[idx]) for idx in indices if idx < len(ml_des.features)]
    print(f"Cluster {cluster_label}: {feature_info_des}")

final_features_des = set()

for cluster_idx in sorted(clustered_feat_names_des.keys()):
    features_in_cluster_des = clustered_feat_names_des[cluster_idx]
    if isinstance(features_in_cluster_des, list):
        final_features_des.add(features_in_cluster_des[-1])
    else:
        final_features_des.add(features_in_cluster_des)

final_features_list_des = list(final_features_des)

data_1_des = pd.concat([data_des[final_features_list_des], y_des_df], axis=1)
"""descriptor data"""


"""morgan data"""
feature_indices_mor = ml_mor.ntri_feat_indices

features_mor = np.array(ml_mor.features)

clustered_feat_names_mor = ml_mor.clustered_feat_names

for cluster_idx in sorted(clustered_feat_names_mor.keys()):
    print(f"Cluster {cluster_idx}: {clustered_feat_names_mor[cluster_idx]}")

for cluster_label, indices in ml_mor.ntri_feat_cluster.items():
    feature_info_mor = [(idx, ml_mor.features[idx]) for idx in indices if idx < len(ml_mor.features)]
    print(f"Cluster {cluster_label}: {feature_info_mor}")

final_features_mor = set()

for cluster_idx in sorted(clustered_feat_names_mor.keys()):
    features_in_cluster_mor = clustered_feat_names_mor[cluster_idx]
    if isinstance(features_in_cluster_mor, list):
        final_features_mor.add(features_in_cluster_mor[-1])
    else:
        final_features_mor.add(features_in_cluster_mor)

final_features_list_mor = list(final_features_mor)

data_1_mor = pd.concat([data_mor[final_features_list_mor], y_mor_df], axis=1)
"""morgan data"""
#%%
# The newly defined static "ml_1" captures the clustering features of the important features
"""literature data"""
features_1_lit = data_lit[final_features_list_lit].columns.tolist()
ml_1_lit = MachineLearning_(
    work_dir=work_dir_lit,
    data=data_1_lit,
    features=features_1_lit,
    target='logD',
    estimator=model,
    #xscaler=scaler
)

ml_1_lit.preprocess()

ml_1_lit.train_test_split()

ml_1_lit.quickly_feature_selection()

ml_1_lit.recursive_feature_selection()

# Get the feature after recursive feature selection
essential_features_lit = ml_1_lit.features
data_essential_feat_lit = pd.concat([data_1_lit[essential_features_lit]], axis=1)

"""literature data"""


"""descriptor data"""
features_1_des = data_des[final_features_list_des].columns.tolist()
ml_1_des = MachineLearning_(
    work_dir=work_dir_des,
    data=data_1_des,
    features=features_1_des,
    target='logD',
    estimator=model,
    #xscaler=scaler
)

ml_1_des.preprocess()

ml_1_des.train_test_split()

ml_1_des.quickly_feature_selection()

ml_1_des.recursive_feature_selection()

essential_features_des = ml_1_des.features
data_essential_feat_des = pd.concat([data_1_des[essential_features_des]], axis=1)

"""descriptor data"""


"""morgan data"""
features_1_mor = data_mor[final_features_list_mor].columns.tolist()
ml_1_mor = MachineLearning_(
    work_dir=work_dir_mor,
    data=data_1_mor,
    features=features_1_mor,
    target='logD',
    estimator=model,
    #xscaler=scaler
)

ml_1_mor.preprocess()

ml_1_mor.train_test_split()

ml_1_mor.quickly_feature_selection()

ml_1_mor.recursive_feature_selection()

essential_features_mor = ml_1_mor.features
data_essential_feat_mor = pd.concat([data_1_mor[essential_features_mor]], axis=1)

"""morgan data"""

#%%
# The model is trained using the features after feature engineering

"""literature data"""
X_engineered_lit = data_essential_feat_lit
engineered_feature_names_lit = list(data_essential_feat_lit.columns)
y_model_lit = y_lit_df
X_train_lit, X_val_lit, y_train_lit, y_val_lit = train_test_split(X_engineered_lit, y_model_lit,
                                                                  test_size=0.2, random_state=42)
def objective_lit(trial):
    params_lit = {
        'objective': 'reg:squarederror',
        'eval_metric': 'mae',
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 1e2, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 1e2, log=True)
    }

    model_lit = XGBRegressor(**params_lit)
    kf_lit = KFold(n_splits=5, shuffle=True, random_state=42)
    # Debug using error_score='raise'
    scores_lit = cross_val_score(model_lit, X_train_lit, y_train_lit,
                             cv=kf_lit, scoring='neg_mean_absolute_error', error_score='raise')
    return -np.mean(scores_lit)

# Create the optimizer and optimize the objective function
study_lit = optuna.create_study(direction='minimize')
study_lit.optimize(objective_lit, n_trials=100)


"""literature data"""


"""descriptor data"""
X_engineered_des = data_essential_feat_des
engineered_feature_names_des = list(data_essential_feat_des.columns)
y_model_des = y_des_df
X_train_des, X_val_des, y_train_des, y_val_des = train_test_split(X_engineered_des, y_model_des,
                                                                  test_size=0.2, random_state=42)
def objective_des(trial):
    params_des = {
        'objective': 'reg:squarederror',
        'eval_metric': 'mae',
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 1e2, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 1e2, log=True)
    }

    model_des = XGBRegressor(**params_des)
    kf_des = KFold(n_splits=5, shuffle=True, random_state=42)
    scores_des = cross_val_score(model_des, X_train_des, y_train_des,
                             cv=kf_des, scoring='neg_mean_absolute_error', error_score='raise')
    return -np.mean(scores_des)

study_des = optuna.create_study(direction='minimize')
study_des.optimize(objective_des, n_trials=100)

"""descriptor data"""


"""morgan data"""
X_engineered_mor = data_essential_feat_mor
engineered_feature_names_mor = list(data_essential_feat_mor.columns)
y_model_mor = y_mor_df
X_train_mor, X_val_mor, y_train_mor, y_val_mor = train_test_split(X_engineered_mor, y_model_mor,
                                                                  test_size=0.2, random_state=42)
def objective_mor(trial):
    params_mor = {
        'objective': 'reg:squarederror',
        'eval_metric': 'mae',
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 1e2, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 1e2, log=True)
    }

    model_mor = XGBRegressor(**params_mor)
    kf_mor = KFold(n_splits=5, shuffle=True, random_state=42)
    scores_mor = cross_val_score(model_mor, X_train_mor, y_train_mor,
                             cv=kf_mor, scoring='neg_mean_absolute_error', error_score='raise')
    return -np.mean(scores_mor)

# Create the optimizer and optimize the objective function
study_mor = optuna.create_study(direction='minimize')
study_mor.optimize(objective_mor, n_trials=100)

"""morgan data"""


# Output optimal parameters and target values
"""literature data"""
print(f"literature Best parameters: {study_lit.best_params}")
print(f"literature Best score: {study_lit.best_value:.4f}")
"""literature data"""

"""descriptor data"""
print(f"descriptor Best parameters: {study_des.best_params}")
print(f"descripor Best score: {study_des.best_value:.4f}")
"""descriptor data"""

"""morgan data"""
print(f"morgan Best parameters: {study_mor.best_params}")
print(f"morgan Best score: {study_mor.best_value:.4f}")
"""morgan data"""

#%%

# Retrain the model with the best parameters
"""literature data"""
best_params_lit = study_lit.best_params
best_model_lit = XGBRegressor(**best_params_lit)
best_model_lit.fit(X_train_lit, y_train_lit)
"""literature data"""

"""descriptor data"""
best_params_des = study_des.best_params
best_model_des = XGBRegressor(**best_params_des)
best_model_des.fit(X_train_des, y_train_des)
"""descriptor data"""

"""morgan data"""
best_params_mor = study_mor.best_params
best_model_mor = XGBRegressor(**best_params_mor)
best_model_mor.fit(X_train_mor, y_train_mor)
"""morgan data"""


# Get feature importance
# Print feature importance and name

"""literature data"""
importance_lit = best_model_lit.feature_importances_

for i, v in enumerate(importance_lit):
    print(f'Feature {engineered_feature_names_lit[i]}: {v}')
"""literature data"""

print("\n")

"""descriptor data"""
importance_des = best_model_des.feature_importances_

for i, v in enumerate(importance_des):
    print(f'Feature {engineered_feature_names_des[i]}: {v}')
"""descriptor data"""

print("\n")

"""morgan data"""
importance_mor = best_model_mor.feature_importances_

for i, v in enumerate(importance_mor):
    print(f'Feature {engineered_feature_names_mor[i]}: {v}')
"""morgan data"""



# Make predictions in both sets
# Calculate evaluation indicators
# Output evaluation index results

"""literature data"""
train_predictions_lit = best_model_lit.predict(X_train_lit)
val_predictions_lit = best_model_lit.predict(X_val_lit)

train_mae_lit = mean_absolute_error(y_train_lit, train_predictions_lit)
train_rmse_lit = np.sqrt(mean_squared_error(y_train_lit, train_predictions_lit))
train_r2_lit = r2_score(y_train_lit, train_predictions_lit)

val_mae_lit = mean_absolute_error(y_val_lit, val_predictions_lit)
val_rmse_lit = np.sqrt(mean_squared_error(y_val_lit, val_predictions_lit))
val_r2_lit = r2_score(y_val_lit, val_predictions_lit)
"""literature data"""

"""descriptor data"""
train_predictions_des = best_model_des.predict(X_train_des)
val_predictions_des = best_model_des.predict(X_val_des)

train_mae_des = mean_absolute_error(y_train_des, train_predictions_des)
train_rmse_des = np.sqrt(mean_squared_error(y_train_des, train_predictions_des))
train_r2_des = r2_score(y_train_des, train_predictions_des)

val_mae_des = mean_absolute_error(y_val_des, val_predictions_des)
val_rmse_des = np.sqrt(mean_squared_error(y_val_des, val_predictions_des))
val_r2_des = r2_score(y_val_des, val_predictions_des)
"""descriptor data"""

"""morgan data"""
train_predictions_mor = best_model_mor.predict(X_train_mor)
val_predictions_mor = best_model_mor.predict(X_val_mor)

train_mae_mor = mean_absolute_error(y_train_mor, train_predictions_mor)
train_rmse_mor = np.sqrt(mean_squared_error(y_train_mor, train_predictions_mor))
train_r2_mor = r2_score(y_train_mor, train_predictions_mor)

val_mae_mor = mean_absolute_error(y_val_mor, val_predictions_mor)
val_rmse_mor = np.sqrt(mean_squared_error(y_val_mor, val_predictions_mor))
val_r2_mor = r2_score(y_val_mor, val_predictions_mor)
"""morgan data"""

#%%

# Output evaluation index results
print(f"literature train MAE: {train_mae_lit:.4f}")
print(f"literature train RMSE: {train_rmse_lit:.4f}")
print(f"literature train R^2: {train_r2_lit:.4f}")
print(f"literature Validation MAE: {val_mae_lit:.4f}")
print(f"literature Validation RMSE: {val_rmse_lit:.4f}")
print(f"literature Validation R^2: {val_r2_lit:.4f}")

print("\n")

print(f"descriptor train MAE: {train_mae_des:.4f}")
print(f"descriptor train RMSE: {train_rmse_des:.4f}")
print(f"descriptor train R^2: {train_r2_des:.4f}")
print(f"descriptor Validation MAE: {val_mae_des:.4f}")
print(f"descriptor Validation RMSE: {val_rmse_des:.4f}")
print(f"descriptor Validation R^2: {val_r2_des:.4f}")

print("\n")

print(f"morgan train MAE: {train_mae_mor:.4f}")
print(f"morgan train RMSE: {train_rmse_mor:.4f}")
print(f"morgan train R^2: {train_r2_mor:.4f}")
print(f"morgan Validation MAE: {val_mae_mor:.4f}")
print(f"morgan Validation RMSE: {val_rmse_mor:.4f}")
print(f"morgan Validation R^2: {val_r2_mor:.4f}")

#%%
"""Now plot scatter plots for separate data sets and observe the state"""

"""literature data"""
# Training set
# Create graphics and axis objects

fig, ax = plt.subplots(figsize=(10, 6))
plt.scatter(y_train_lit, train_predictions_lit, alpha=0.7, marker='P')  # Draw scatter plots
plt.plot([y_train_lit.min(), y_train_lit.max()], [y_train_lit.min(), y_train_lit.max()], color='lightcoral',
         linestyle='--', linewidth=2)  # Draw a 45 degree reference line
# Set the axis border style uniformly
for spine in ax.spines.values():
    spine.set_linewidth(2)   # Border width
    spine.set_color('black') # Boundary color
    spine.set_linestyle('-')# Boundary style

# Set the thickness of the axis scale
ax.tick_params(axis='both', which='major', width=2, labelsize=12)
ax.tick_params(axis='both', which='minor', width=1, labelsize=12)

# Set the axis label
ax.set_xlabel('Actual value of lit', fontsize=18)
ax.set_ylabel('True value of lit', fontsize=18)
plt.title("Train", fontsize=18)  # Set the title

# Add text annotations for evaluation indicators
plt.text(0.05, 0.90, f'$R^2: {train_r2_lit:.4f}$', transform=plt.gca().transAxes,
         fontsize=20, verticalalignment='top')

plt.show()  # Display graphics


# verification set

fig, ax = plt.subplots(figsize=(10, 6))
plt.scatter(y_val_lit, val_predictions_lit, alpha=0.7, marker='P')
plt.plot([y_val_lit.min(), y_val_lit.max()], [y_val_lit.min(), y_val_lit.max()], color='lightcoral',
         linestyle='--', linewidth=2)

for spine in ax.spines.values():
    spine.set_linewidth(2)
    spine.set_color('black')
    spine.set_linestyle('-')

ax.tick_params(axis='both', which='major', width=2, labelsize=12)
ax.tick_params(axis='both', which='minor', width=1, labelsize=12)

ax.set_xlabel('Actual value of lit', fontsize=18)
ax.set_ylabel('True value of lit', fontsize=18)
plt.title("Val", fontsize=18)

plt.text(0.05, 0.90, f'$R^2: {val_r2_lit:.4f}$', transform=plt.gca().transAxes,
         fontsize=20, verticalalignment='top')

plt.show()

"""literature data"""

"""descriptor data"""

# Training set
fig, ax = plt.subplots(figsize=(10, 6))
plt.scatter(y_train_des, train_predictions_des, alpha=0.7, marker='P')
plt.plot([y_train_des.min(), y_train_des.max()], [y_train_des.min(), y_train_des.max()], color='lightcoral',
         linestyle='--', linewidth=2)

for spine in ax.spines.values():
    spine.set_linewidth(2)
    spine.set_color('black')
    spine.set_linestyle('-')

ax.tick_params(axis='both', which='major', width=2, labelsize=12)
ax.tick_params(axis='both', which='minor', width=1, labelsize=12)

ax.set_xlabel('Actual value of des', fontsize=18)
ax.set_ylabel('True value of des', fontsize=18)
plt.title("Train", fontsize=18)

plt.text(0.05, 0.90, f'$R^2: {train_r2_des:.4f}$', transform=plt.gca().transAxes,
         fontsize=20, verticalalignment='top')

plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
plt.scatter(y_val_des, val_predictions_des, alpha=0.7, marker='P')
plt.plot([y_val_des.min(), y_val_des.max()], [y_val_des.min(), y_val_des.max()], color='lightcoral',
         linestyle='--', linewidth=2)

for spine in ax.spines.values():
    spine.set_linewidth(2)
    spine.set_color('black')
    spine.set_linestyle('-')

ax.tick_params(axis='both', which='major', width=2, labelsize=12)
ax.tick_params(axis='both', which='minor', width=1, labelsize=12)

ax.set_xlabel('Actual value of des', fontsize=18)
ax.set_ylabel('True value of des', fontsize=18)
plt.title("Val", fontsize=18)

plt.text(0.05, 0.90, f'$R^2: {val_r2_des:.4f}$', transform=plt.gca().transAxes,
         fontsize=20, verticalalignment='top')

plt.show()

"""descriptor data"""

"""morgan data"""

# Training set
fig, ax = plt.subplots(figsize=(10, 6))
plt.scatter(y_train_mor, train_predictions_mor, alpha=0.7, marker='P')
plt.plot([y_train_mor.min(), y_train_mor.max()], [y_train_mor.min(), y_train_mor.max()], color='lightcoral',
         linestyle='--', linewidth=2)

for spine in ax.spines.values():
    spine.set_linewidth(2)
    spine.set_color('black')
    spine.set_linestyle('-')

ax.tick_params(axis='both', which='major', width=2, labelsize=12)
ax.tick_params(axis='both', which='minor', width=1, labelsize=12)

# Set the axis label
ax.set_xlabel('Actual value of mor', fontsize=18)
ax.set_ylabel('True value of mor', fontsize=18)
plt.title("Train", fontsize=18)

plt.text(0.05, 0.90, f'$R^2: {train_r2_mor:.4f}$', transform=plt.gca().transAxes,
         fontsize=20, verticalalignment='top')

plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
plt.scatter(y_val_mor, val_predictions_mor, alpha=0.7, marker='P')
plt.plot([y_val_mor.min(), y_val_mor.max()], [y_val_mor.min(), y_val_mor.max()], color='lightcoral',
         linestyle='--', linewidth=2)

for spine in ax.spines.values():
    spine.set_linewidth(2)
    spine.set_color('black')
    spine.set_linestyle('-')

ax.tick_params(axis='both', which='major', width=2, labelsize=12)
ax.tick_params(axis='both', which='minor', width=1, labelsize=12)

ax.set_xlabel('Actual value of mor', fontsize=18)
ax.set_ylabel('True value of mor', fontsize=18)
plt.title("Val", fontsize=18)

plt.text(0.05, 0.90, f'$R^2: {val_r2_mor:.4f}$', transform=plt.gca().transAxes,
         fontsize=20, verticalalignment='top')

plt.show()

"""morgan data"""



"""At this point, the work of each independent module has been completed, 
    and now the important features in the literature are integrated into other data sets 
        to observe the impact on the model performance"""

#%%
X_engineered_1_des = np.concatenate((data_essential_feat_des, data_essential_feat_lit), axis=1)
engineered_feature_names_1_des = list(data_essential_feat_des.columns) + list(data_essential_feat_lit.columns)
y_1_des = y_des_df

X_train_1_des, X_val_1_des, y_train_1_des, y_val_1_des = train_test_split(
    X_engineered_1_des, y_1_des, test_size=0.2, random_state=42)

def objective_1_des(trial):
    params_1_des = {
        'objective': 'reg:squarederror',
        'eval_metric': 'mae',
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 1e2, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 1e2, log=True)
    }

    model_1_des = XGBRegressor(**params_1_des)
    kf_1_des = KFold(n_splits=5, shuffle=True, random_state=42)
    scores_1_des = cross_val_score(model_1_des, X_train_1_des, y_train_1_des,
                             cv=kf_1_des, scoring='neg_mean_absolute_error', error_score='raise')
    return -np.mean(scores_1_des)

study_1_des = optuna.create_study(direction='minimize')
study_1_des.optimize(objective_1_des, n_trials=100)

print(f"newly descriptor Best parameters: {study_1_des.best_params}")
print(f"newly descriptor Best score: {study_1_des.best_value:.4f}")

best_params_1_des = study_1_des.best_params
best_model_1_des = XGBRegressor(**best_params_1_des)
best_model_1_des.fit(X_train_1_des, y_train_1_des)

importance_1_des = best_model_1_des.feature_importances_

for i, v in enumerate(importance_1_des):
    print(f'Feature {engineered_feature_names_1_des[i]}: {v}')

train_predictions_1_des = best_model_1_des.predict(X_train_1_des)
val_predictions_1_des = best_model_1_des.predict(X_val_1_des)

train_mae_1_des = mean_absolute_error(y_train_1_des, train_predictions_1_des)
train_rmse_1_des = np.sqrt(mean_squared_error(y_train_1_des, train_predictions_1_des))
train_r2_1_des = r2_score(y_train_1_des, train_predictions_1_des)

val_mae_1_des = mean_absolute_error(y_val_1_des, val_predictions_1_des)
val_rmse_1_des = np.sqrt(mean_squared_error(y_val_1_des, val_predictions_1_des))
val_r2_1_des = r2_score(y_val_1_des, val_predictions_1_des)

#%%
print(f"newly descriptor train MAE: {train_mae_1_des:.4f}")
print(f"newly descriptor train RMSE: {train_rmse_1_des:.4f}")
print(f"newly descriptor train R^2: {train_r2_1_des:.4f}")
print(f"newly descriptor Validation MAE: {val_mae_1_des:.4f}")
print(f"newly descriptor Validation RMSE: {val_rmse_1_des:.4f}")
print(f"newly descriptor Validation R^2: {val_r2_1_des:.4f}")
#%%
X_engineered_1_mor = np.concatenate((data_essential_feat_mor, data_essential_feat_lit), axis=1)
engineered_feature_names_1_mor = list(data_essential_feat_mor.columns) + list(data_essential_feat_lit.columns)
y_1_mor = y_mor_df

X_train_1_mor, X_val_1_mor, y_train_1_mor, y_val_1_mor = train_test_split(
    X_engineered_1_mor, y_1_mor, test_size=0.2, random_state=42)

def objective_1_mor(trial):
    params_1_mor = {
        'objective': 'reg:squarederror',
        'eval_metric': 'mae',
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 1e2, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 1e2, log=True)
    }

    model_1_mor = XGBRegressor(**params_1_mor)
    kf_1_mor = KFold(n_splits=5, shuffle=True, random_state=42)
    scores_1_mor = cross_val_score(model_1_mor, X_train_1_mor, y_train_1_mor,
                             cv=kf_1_mor, scoring='neg_mean_absolute_error', error_score='raise')
    return -np.mean(scores_1_mor)

study_1_mor = optuna.create_study(direction='minimize')
study_1_mor.optimize(objective_1_mor, n_trials=100)

print(f"newly morgan Best parameters: {study_1_mor.best_params}")
print(f"newly morgan Best score: {study_1_mor.best_value:.4f}")

best_params_1_mor = study_1_mor.best_params
best_model_1_mor = XGBRegressor(**best_params_1_mor)
best_model_1_mor.fit(X_train_1_mor, y_train_1_mor)

importance_1_mor = best_model_1_mor.feature_importances_

for i, v in enumerate(importance_1_mor):
    print(f'Feature {engineered_feature_names_1_mor[i]}: {v}')

train_predictions_1_mor = best_model_1_mor.predict(X_train_1_mor)
val_predictions_1_mor = best_model_1_mor.predict(X_val_1_mor)

train_mae_1_mor = mean_absolute_error(y_train_1_mor, train_predictions_1_mor)
train_rmse_1_mor = np.sqrt(mean_squared_error(y_train_1_mor, train_predictions_1_mor))
train_r2_1_mor = r2_score(y_train_1_mor, train_predictions_1_mor)

val_mae_1_mor = mean_absolute_error(y_val_1_mor, val_predictions_1_mor)
val_rmse_1_mor = np.sqrt(mean_squared_error(y_val_1_mor, val_predictions_1_mor))
val_r2_1_mor = r2_score(y_val_1_mor, val_predictions_1_mor)

#%%
print(f"newly morgan train MAE: {train_mae_1_mor:.4f}")
print(f"newly morgan train RMSE: {train_rmse_1_mor:.4f}")
print(f"newly morgan train R^2: {train_r2_1_mor:.4f}")
print(f"newly morgan Validation MAE: {val_mae_1_mor:.4f}")
print(f"newly morgan Validation RMSE: {val_rmse_1_mor:.4f}")
print(f"newly morgan Validation R^2: {val_r2_1_mor:.4f}")

#%%
X_engineered_mix = np.concatenate((data_essential_feat_des, data_essential_feat_mor, data_essential_feat_lit), axis=1)
engineered_feature_names_mix = (list(data_essential_feat_des.columns)
                                  + list(data_essential_feat_mor.columns)
                                  + list(data_essential_feat_lit.columns))
y_mix = y_mor_df

X_train_mix, X_val_mix, y_train_mix, y_val_mix = train_test_split(
    X_engineered_mix, y_mix, test_size=0.2, random_state=42)

def objective_mix(trial):
    params_mix = {
        'objective': 'reg:squarederror',
        'eval_metric': 'mae',
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 1e2, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 1e2, log=True)
    }

    model_mix = XGBRegressor(**params_mix)
    kf_mix = KFold(n_splits=5, shuffle=True, random_state=42)
    scores_mix = cross_val_score(model_mix, X_train_mix, y_train_mix,
                             cv=kf_mix, scoring='neg_mean_absolute_error', error_score='raise')
    return -np.mean(scores_mix)

study_mix = optuna.create_study(direction='minimize')
study_mix.optimize(objective_mix, n_trials=100)

print(f"descriptor Best parameters: {study_mix.best_params}")
print(f"descriptor Best score: {study_mix.best_value:.4f}")

best_params_mix = study_mix.best_params
best_model_mix = XGBRegressor(**best_params_mix)
best_model_mix.fit(X_train_mix, y_train_mix)

importance_mix = best_model_mix.feature_importances_
for i, v in enumerate(importance_mix):
    print(f'Feature {engineered_feature_names_mix[i]}: {v}')

#%%
train_predictions_mix = best_model_mix.predict(X_train_mix)
val_predictions_mix = best_model_mix.predict(X_val_mix)

train_mae_mix = mean_absolute_error(y_train_mix, train_predictions_mix)
train_rmse_mix = np.sqrt(mean_squared_error(y_train_mix, train_predictions_mix))
train_r2_mix = r2_score(y_train_mix, train_predictions_mix)

val_mae_mix = mean_absolute_error(y_val_mix, val_predictions_mix)
val_rmse_mix = np.sqrt(mean_squared_error(y_val_mix, val_predictions_mix))
val_r2_mix = r2_score(y_val_mix, val_predictions_mix)

#%%
print(f"train MAE: {train_mae_mix:.4f}")
print(f"train RMSE: {train_rmse_mix:.4f}")
print(f"train R^2: {train_r2_mix:.4f}")
print(f"Validation MAE: {val_mae_mix:.4f}")
print(f"Validation RMSE: {val_rmse_mix:.4f}")
print(f"Validation R^2: {val_r2_mix:.4f}")

#%%
indices = np.arange(len(X_engineered_mix))
X_train_indices, X_val_indices, y_train_mix, y_val_mix = train_test_split(
    indices, y_mix, test_size=0.2, random_state=42
)

# 2. 读取包含SMILES的原始Excel文件（需指定路径） /home/kylewu/hotpot/examples/U_logD/
smiles_excel_path = "/home/kylewu/hotpot/examples/U_logD/data/shuffled_total.xlsx"  # 请替换为实际路径
smiles_df = pd.read_excel(smiles_excel_path, engine="openpyxl")

# 3. 提取训练集对应的SMILES
# 确保SMILES数据的顺序与X_engineered_mix完全一致
train_smiles = smiles_df.iloc[X_train_indices]["lig_SMI"]

# 4. 保存为新的Excel文件
output_path = "/home/kylewu/hotpot/examples/U_logD/ML_newlig/train_smiles.xlsx"  # 请替换为实际保存路径
train_smiles.to_excel(output_path, index=False, engine="openpyxl")

print(f"训练集SMILES已保存至：{output_path}")

#%%
import os
import joblib
from sklearn.preprocessing import StandardScaler

# 定义保存路径
save_dir = "/home/kylewu/hotpot/examples/U_logD/model_para"  # 保存目录
os.makedirs(save_dir, exist_ok=True)  # 如果目录不存在，创建它

# 保存填充器和标准化器
imputer_lit = SimpleImputer(strategy='mean')
imputer_des = SimpleImputer(strategy='mean')
imputer_mor = SimpleImputer(strategy='mean')

scaler_lit = StandardScaler(with_mean=False)
scaler_lit.fit(X_engineered_lit)
scaler_des = StandardScaler(with_mean=False)
scaler_des.fit(X_engineered_des)
scaler_mor = StandardScaler(with_mean=False)
scaler_mor.fit(X_engineered_mor)

# 保存填充器
joblib.dump(imputer_lit, os.path.join(save_dir, 'imputer_lit.pkl'))
joblib.dump(imputer_des, os.path.join(save_dir, 'imputer_des.pkl'))
joblib.dump(imputer_mor, os.path.join(save_dir, 'imputer_mor.pkl'))

# 保存标准化器
joblib.dump(scaler_lit, os.path.join(save_dir, 'scaler_lit.pkl'))
joblib.dump(scaler_des, os.path.join(save_dir, 'scaler_des.pkl'))
joblib.dump(scaler_mor, os.path.join(save_dir, 'scaler_mor.pkl'))

print(f"填充器和标准化器已保存至：{save_dir}")

# 保存模型
joblib.dump(best_model_lit, os.path.join(save_dir, 'best_model_lit.pkl'))
joblib.dump(best_model_des, os.path.join(save_dir, 'best_model_des.pkl'))
joblib.dump(best_model_mor, os.path.join(save_dir, 'best_model_mor.pkl'))
joblib.dump(best_model_1_des, os.path.join(save_dir, 'best_model_1_des.pkl'))
joblib.dump(best_model_1_mor, os.path.join(save_dir, 'best_model_1_mor.pkl'))
joblib.dump(best_model_mix, os.path.join(save_dir, 'best_model_mix.pkl'))

print(f"所有模型已保存至：{save_dir}")

# 保存特征名称
joblib.dump(engineered_feature_names_lit, os.path.join(save_dir, 'engineered_feature_names_lit.pkl'))
joblib.dump(engineered_feature_names_des, os.path.join(save_dir, 'engineered_feature_names_des.pkl'))
joblib.dump(engineered_feature_names_mor, os.path.join(save_dir, 'engineered_feature_names_mor.pkl'))
joblib.dump(engineered_feature_names_1_des, os.path.join(save_dir, 'engineered_feature_names_1_des.pkl'))
joblib.dump(engineered_feature_names_1_mor, os.path.join(save_dir, 'engineered_feature_names_1_mor.pkl'))
joblib.dump(engineered_feature_names_mix, os.path.join(save_dir, 'engineered_feature_names_mix.pkl'))

print(f"所有特征名称已保存至：{save_dir}")

#%%
"""
模型调用示例代码

# 加载填充器
imputer_lit = joblib.load(os.path.join(save_dir, 'imputer_lit.pkl'))
imputer_des = joblib.load(os.path.join(save_dir, 'imputer_des.pkl'))
imputer_mor = joblib.load(os.path.join(save_dir, 'imputer_mor.pkl'))

# 加载标准化器
scaler_lit = joblib.load(os.path.join(save_dir, 'scaler_lit.pkl'))
scaler_des = joblib.load(os.path.join(save_dir, 'scaler_des.pkl'))
scaler_mor = joblib.load(os.path.join(save_dir, 'scaler_mor.pkl'))

# 加载模型
best_model_lit = joblib.load(os.path.join(save_dir, 'best_model_lit.pkl'))
best_model_des = joblib.load(os.path.join(save_dir, 'best_model_des.pkl'))
best_model_mor = joblib.load(os.path.join(save_dir, 'best_model_mor.pkl'))
best_model_1_des = joblib.load(os.path.join(save_dir, 'best_model_1_des.pkl'))
best_model_1_mor = joblib.load(os.path.join(save_dir, 'best_model_1_mor.pkl'))
best_model_mix = joblib.load(os.path.join(save_dir, 'best_model_mix.pkl'))

# 加载特征名称
engineered_feature_names_lit = joblib.load(os.path.join(save_dir, 'engineered_feature_names_lit.pkl'))
engineered_feature_names_des = joblib.load(os.path.join(save_dir, 'engineered_feature_names_des.pkl'))
engineered_feature_names_mor = joblib.load(os.path.join(save_dir, 'engineered_feature_names_mor.pkl'))
engineered_feature_names_1_des = joblib.load(os.path.join(save_dir, 'engineered_feature_names_1_des.pkl'))
engineered_feature_names_1_mor = joblib.load(os.path.join(save_dir, 'engineered_feature_names_1_mor.pkl'))
engineered_feature_names_mix = joblib.load(os.path.join(save_dir, 'engineered_feature_names_mix.pkl'))

print("所有填充器、标准化器、模型和特征名称已成功加载。")

import pandas as pd

# 加载新数据
new_data_lit = pd.read_csv('/path/to/new_data_lit.csv')
new_data_des = pd.read_csv('/path/to/new_data_des.csv')
new_data_mor = pd.read_csv('/path/to/new_data_mor.csv')

# 确保新数据的特征顺序与训练数据一致
new_data_lit = new_data_lit[engineered_feature_names_lit]
new_data_des = new_data_des[engineered_feature_names_des]
new_data_mor = new_data_mor[engineered_feature_names_mor]

# 对新数据进行标准化处理
new_data_lit_scaled = scaler_lit.transform(new_data_lit)
new_data_des_scaled = scaler_des.transform(new_data_des)
new_data_mor_scaled = scaler_mor.transform(new_data_mor)

# 使用模型进行预测
predictions_lit = best_model_lit.predict(new_data_lit_scaled)
predictions_des = best_model_des.predict(new_data_des_scaled)
predictions_mor = best_model_mor.predict(new_data_mor_scaled)

# 保存预测结果
pd.DataFrame(predictions_lit, columns=['predictions_lit']).to_csv('/path/to/predictions_lit.csv', index=False)
pd.DataFrame(predictions_des, columns=['predictions_des']).to_csv('/path/to/predictions_des.csv', index=False)
pd.DataFrame(predictions_mor, columns=['predictions_mor']).to_csv('/path/to/predictions_mor.csv', index=False)

print("预测结果已保存。")

"""

#%%
# ====================================================
# 可视化部分
# ====================================================

# scatter plots

"""des"""
# Training set

fig, ax = plt.subplots(figsize=(10, 6))
plt.scatter(y_train_1_des, train_predictions_1_des, alpha=0.7, marker='P')
plt.plot([y_train_1_des.min(), y_train_1_des.max()], [y_train_1_des.min(), y_train_1_des.max()], color='lightcoral',
         linestyle='--', linewidth=2)

for spine in ax.spines.values():
    spine.set_linewidth(2)
    spine.set_color('black')
    spine.set_linestyle('-')

ax.tick_params(axis='both', which='major', width=2, labelsize=12)
ax.tick_params(axis='both', which='minor', width=1, labelsize=12)

ax.set_xlabel('Actual value of des', fontsize=18)
ax.set_ylabel('True value of des', fontsize=18)
plt.title("Train", fontsize=18)

plt.text(0.05, 0.90, f'$R^2: {train_r2_1_des:.4f}$', transform=plt.gca().transAxes,
         fontsize=20, verticalalignment='top')

plt.show()

# Verification set
fig, ax = plt.subplots(figsize=(10, 6))
plt.scatter(y_val_1_des, val_predictions_1_des, alpha=0.7, marker='P')
plt.plot([y_val_1_des.min(), y_val_1_des.max()], [y_val_1_des.min(), y_val_1_des.max()], color='lightcoral',
         linestyle='--', linewidth=2)

for spine in ax.spines.values():
    spine.set_linewidth(2)
    spine.set_color('black')
    spine.set_linestyle('-')

ax.tick_params(axis='both', which='major', width=2, labelsize=12)
ax.tick_params(axis='both', which='minor', width=1, labelsize=12)

ax.set_xlabel('Actual value of des', fontsize=18)
ax.set_ylabel('True value of des', fontsize=18)
plt.title("Val", fontsize=18)

plt.text(0.05, 0.90, f'$R^2: {val_r2_1_des:.4f}$', transform=plt.gca().transAxes,
         fontsize=20, verticalalignment='top')

plt.show()

"""des"""

"""mor"""
# Training set
fig, ax = plt.subplots(figsize=(10, 6))
plt.scatter(y_train_1_mor, train_predictions_1_mor, alpha=0.7, marker='P')
plt.plot([y_train_1_mor.min(), y_train_1_mor.max()], [y_train_1_mor.min(), y_train_1_mor.max()], color='lightcoral',
         linestyle='--', linewidth=2)

for spine in ax.spines.values():
    spine.set_linewidth(2)
    spine.set_color('black')
    spine.set_linestyle('-')

ax.tick_params(axis='both', which='major', width=2, labelsize=12)
ax.tick_params(axis='both', which='minor', width=1, labelsize=12)

ax.set_xlabel('Actual value of mor', fontsize=18)
ax.set_ylabel('True value of mor', fontsize=18)
plt.title("Train", fontsize=18)

plt.text(0.05, 0.90, f'$R^2: {train_r2_1_mor:.4f}$', transform=plt.gca().transAxes,
         fontsize=20, verticalalignment='top')

plt.show()

# Verification set
fig, ax = plt.subplots(figsize=(10, 6))
plt.scatter(y_val_1_mor, val_predictions_1_mor, alpha=0.7, marker='P')
plt.plot([y_val_1_mor.min(), y_val_1_mor.max()], [y_val_1_mor.min(), y_val_1_mor.max()], color='lightcoral',
         linestyle='--', linewidth=2)

for spine in ax.spines.values():
    spine.set_linewidth(2)
    spine.set_color('black')
    spine.set_linestyle('-')

ax.tick_params(axis='both', which='major', width=2, labelsize=12)
ax.tick_params(axis='both', which='minor', width=1, labelsize=12)

ax.set_xlabel('Actual value of mor', fontsize=18)
ax.set_ylabel('True value of mor', fontsize=18)
plt.title("Val", fontsize=18)

plt.text(0.05, 0.90, f'$R^2: {val_r2_1_mor:.4f}$', transform=plt.gca().transAxes,
         fontsize=20, verticalalignment='top')

plt.show()

"""mor"""

"""mix"""
# Training set
fig, ax = plt.subplots(figsize=(10, 6))
plt.scatter(y_train_mix, train_predictions_mix, alpha=0.7, marker='P')
plt.plot([y_train_mix.min(), y_train_mix.max()], [y_train_mix.min(), y_train_mix.max()], color='lightcoral',
         linestyle='--', linewidth=2)

for spine in ax.spines.values():
    spine.set_linewidth(2)
    spine.set_color('black')
    spine.set_linestyle('-')

ax.tick_params(axis='both', which='major', width=2, labelsize=12)
ax.tick_params(axis='both', which='minor', width=1, labelsize=12)

ax.set_xlabel('Actual value of mix', fontsize=18)
ax.set_ylabel('True value of mix', fontsize=18)
plt.title("Train", fontsize=18)

plt.text(0.05, 0.90, f'$R^2: {train_r2_mix:.4f}$', transform=plt.gca().transAxes,
         fontsize=20, verticalalignment='top')

plt.show()

# verification set

fig, ax = plt.subplots(figsize=(10, 6))
plt.scatter(y_val_mix, val_predictions_mix, alpha=0.7, marker='P')
plt.plot([y_val_mix.min(), y_val_mix.max()], [y_val_mix.min(), y_val_mix.max()], color='lightcoral',
         linestyle='--', linewidth=2)

for spine in ax.spines.values():
    spine.set_linewidth(2)
    spine.set_color('black')
    spine.set_linestyle('-')

ax.tick_params(axis='both', which='major', width=2, labelsize=12)
ax.tick_params(axis='both', which='minor', width=1, labelsize=12)

ax.set_xlabel('Actual value of mix', fontsize=18)
ax.set_ylabel('True value of mix', fontsize=18)
plt.title("Val", fontsize=18)

plt.text(0.05, 0.90, f'$R^2: {val_r2_mix:.4f}$', transform=plt.gca().transAxes,
         fontsize=20, verticalalignment='top')

plt.show()
"""mix"""

#%%
"""After completing the above work, we will perform a sahp analysis of the fusion feature set"""

# 创建 SHAP 解释器
explainer = shap.TreeExplainer(best_model_mix)

# 计算 SHAP 值
shap_values = explainer.shap_values(X_train_mix)

# 1. 计算每个特征的 SHAP 值的平均值
# 目标变量名称
shap_values_class = shap_values  # 直接使用二维数组

#%%
# 双轴shap可视化：蜂巢图、特征重要性图

fig, ax1 = plt.subplots(figsize=(15, 10), dpi=300) #适当改变分辨率

# 在主图上绘制蜂巢图，并保留热度条
shap.summary_plot(shap_values_class, X_train_mix, feature_names=engineered_feature_names_mix,
                  plot_type="dot", cmap='RdBu', show=False, color_bar=True)
plt.gca().set_position([0.2, 0.2, 0.65, 0.65])  # 调整图表位置，留出右侧空间放热度条

# 获取共享的 y 轴
ax1 = plt.gca()

# 创建共享 y 轴的另一个图，绘制特征贡献图在顶部x轴
ax2 = ax1.twiny()
shap.summary_plot(shap_values_class, X_train_mix, feature_names=engineered_feature_names_mix,
                  plot_type="bar", show=False)
plt.gca().set_position([0.2, 0.2, 0.65, 0.65])  # 调整图表位置，与蜂巢图对齐

# 在顶部 X 轴添加一条横线
ax2.axhline(y=11, color='k', linestyle='-', linewidth=1.5)  # 注意y值应该对应顶部

# 调整透明度
bars = ax2.patches  # 获取所有的柱状图对象
for bar in bars:
    bar.set_facecolor('gray')  # 设置颜色
    bar.set_alpha(0.4)  # 设置透明度

# 设置两个x轴的标签
ax1.set_xlabel('Shapley Value Contribution (Bee Swarm)', fontsize=10)
ax2.set_xlabel('Mean Shapley Value (Feature Importance)', fontsize=10)

# 移动顶部的 X 轴，避免与底部 X 轴重叠
ax2.xaxis.set_label_position('top') # 将标签移动到顶部
ax2.xaxis.tick_top()  # 将刻度也移动到顶部

# 设置y轴标签
ax1.set_ylabel('Features', fontsize=12)

# 调整图表的边距
plt.subplots_adjust(top=0.9, bottom=0.15, left=0.1, right=0.9, hspace=0.4, wspace=0.4)

plt.tight_layout()
# plt.savefig("SHAP_combined_with_top_line_corrected.pdf", format='pdf', bbox_inches='tight')
plt.show()

#%%
# Y轴为序号

# 双轴shap可视化：蜂巢图、特征重要性图
fig, ax1 = plt.subplots(figsize=(15, 10), dpi=300)  # 适当改变分辨率

# 在主图上绘制蜂巢图，并保留热度条
shap.summary_plot(shap_values_class, X_train_mix, feature_names=engineered_feature_names_mix,
                  plot_type="dot", cmap='RdBu', show=False, color_bar=True)
plt.gca().set_position([0.2, 0.2, 0.65, 0.65])  # 调整图表位置，留出右侧空间放热度条

# 获取共享的 y 轴
ax1 = plt.gca()

# 创建共享 y 轴的另一个图，绘制特征贡献图在顶部x轴
ax2 = ax1.twiny()
shap.summary_plot(shap_values_class, X_train_mix, feature_names=engineered_feature_names_mix,
                  plot_type="bar", show=False)
plt.gca().set_position([0.2, 0.2, 0.65, 0.65])  # 调整图表位置，与蜂巢图对齐

# 在顶部 X 轴添加一条横线
ax2.axhline(y=20, color='k', linestyle='-', linewidth=1.5)  # 注意y值应该对应顶部

# 调整透明度
bars = ax2.patches  # 获取所有的柱状图对象
for bar in bars:
    bar.set_facecolor('gray')  # 设置颜色
    bar.set_alpha(0.4)  # 设置透明度

# 设置两个x轴的标签
ax1.set_xlabel('Shapley Value Contribution (Bee Swarm)', fontsize=10)
ax2.set_xlabel('Mean Shapley Value (Feature Importance)', fontsize=10)

# 移动顶部的 X 轴，避免与底部 X 轴重叠
ax2.xaxis.set_label_position('top')  # 将标签移动到顶部
ax2.xaxis.tick_top()  # 将刻度也移动到顶部

# 设置y轴标签
ax1.set_ylabel('Features index', fontsize=12)

# 调整图表的边距
plt.subplots_adjust(top=0.9, bottom=0.15, left=0.1, right=0.9, hspace=0.4, wspace=0.4)

# 获取 y 轴标签（特征的索引）
num_features = 11 # 只显示前 20 个特征
y_labels = list(range(num_features, 0, -1))  # 倒序显示：从 20 到 1

# 设置y轴为数字索引
ax1.set_yticks(list(range(num_features)))  # 设置前20个特征的索引
ax1.set_yticklabels(y_labels)  # 设置倒序的标签，从 20 到 1

# 展示图表
plt.tight_layout()
plt.show()

#%%
# 三线表部分

# 计算 SHAP 值的重要性（绝对值的平均值）
mean_shap_values = np.abs(shap_values_class).mean(axis=0)

# 获取前20个重要的特征索引
top_20_feature_indices = np.argsort(mean_shap_values)[-20:][::-1]  # 从大到小排序，取前20个

# 获取对应的特征名称
top_20_feature_names = [engineered_feature_names_mix[i] for i in top_20_feature_indices]

# 创建一个 DataFrame，显示序号和特征名称
table_data = pd.DataFrame({
    'Index': range(1, 12),  # 从 1 到 20
    'Feature': top_20_feature_names
})

output_path_shaptable = "/home/kylewu/hotpot/examples/U_logD/shap/shap_table.xlsx"  # 请替换为实际保存路径
table_data.to_excel(output_path_shaptable, index=False, engine="openpyxl")

#%%

# 配置输出目录
output_dir_train = "/home/kylewu/hotpot/examples/U_logD/plot/train"
os.makedirs(output_dir_train, exist_ok=True)

# 训练集
datasets_train = [
    ("lit", y_train_lit, train_predictions_lit),
    ("des", y_train_des, train_predictions_des),
    ("mor", y_train_mor, train_predictions_mor),
    ("des_lit", y_train_1_des, train_predictions_1_des),
    ("mor_lit", y_train_1_mor, train_predictions_1_mor),
    ("mix", y_train_mix, train_predictions_mix)
]

for dataset_name, y_train, train_pre in datasets_train:
    df = pd.DataFrame({
        "y_train": y_train.to_numpy().ravel(),  # 兼容 DataFrame 和 ndarray
        "train_pre": train_pre
    })
    save_path = os.path.join(output_dir_train, f"{dataset_name}_train.xlsx")
    df.to_excel(save_path, index=False, engine="openpyxl")
    print(f"{dataset_name}_train 训练集结果已保存至：{save_path}")

# 配置验证集输出目录
output_dir_val = "/home/kylewu/hotpot/examples/U_logD/plot/val"
os.makedirs(output_dir_val, exist_ok=True)

# 验证集
datasets_val = [
    ("lit", y_val_lit, val_predictions_lit),
    ("des", y_val_des, val_predictions_des),
    ("mor", y_val_mor, val_predictions_mor),
    ("des_lit", y_val_1_des, val_predictions_1_des),
    ("mor_lit", y_val_1_mor, val_predictions_1_mor),
    ("mix", y_val_mix, val_predictions_mix)
]

for dataset_name, y_val, val_pre in datasets_val:
    df = pd.DataFrame({
        "y_val": y_val.to_numpy().ravel(),
        "val_pre": val_pre
    })
    save_path = os.path.join(output_dir_val, f"{dataset_name}_val.xlsx")
    df.to_excel(save_path, index=False, engine="openpyxl")
    print(f"{dataset_name}_val 验证集结果已保存至：{save_path}")

print("\n所有训练集和验证集结果保存完成！")

#%%
# 创建模型指标列表
models = [
    {"Model": "Literature", "Train MAE": train_mae_lit, "Train RMSE": train_rmse_lit, "Train R²": train_r2_lit,
     "Val MAE": val_mae_lit, "Val RMSE": val_rmse_lit, "Val R²": val_r2_lit},

    {"Model": "Descriptor", "Train MAE": train_mae_des, "Train RMSE": train_rmse_des, "Train R²": train_r2_des,
     "Val MAE": val_mae_des, "Val RMSE": val_rmse_des, "Val R²": val_r2_des},

    {"Model": "Morgan", "Train MAE": train_mae_mor, "Train RMSE": train_rmse_mor, "Train R²": train_r2_mor,
     "Val MAE": val_mae_mor, "Val RMSE": val_rmse_mor, "Val R²": val_r2_mor},

    {"Model": "Descriptor + Literature", "Train MAE": train_mae_1_des, "Train RMSE": train_rmse_1_des,
     "Train R²": train_r2_1_des,
     "Val MAE": val_mae_1_des, "Val RMSE": val_rmse_1_des, "Val R²": val_r2_1_des},

    {"Model": "Morgan + Literature", "Train MAE": train_mae_1_mor, "Train RMSE": train_rmse_1_mor,
     "Train R²": train_r2_1_mor,
     "Val MAE": val_mae_1_mor, "Val RMSE": val_rmse_1_mor, "Val R²": val_r2_1_mor},

    {"Model": "Mix (Desc+Mor+Lit)", "Train MAE": train_mae_mix, "Train RMSE": train_rmse_mix, "Train R²": train_r2_mix,
     "Val MAE": val_mae_mix, "Val RMSE": val_rmse_mix, "Val R²": val_r2_mix}
]

# 直接使用 pd.DataFrame 创建
metrics_summary = pd.DataFrame(models)

# 指定保存路径
output_path = "/home/kylewu/hotpot/examples/U_logD/plot/summary.xlsx"

# 保存到 Excel
metrics_summary.to_excel(output_path, index=False, float_format="%.4f")
print(f"评估指标已保存至：{output_path}")

#%%
import os
import optuna

# 定义保存路径
base_dir = "/home/kylewu/hotpot/examples/U_logD/bayes_plot"
os.makedirs(base_dir, exist_ok=True)


# 定义保存图表的函数
def save_optuna_plots(study, study_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # 1. 优化历史图（Optimization History）
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image(os.path.join(save_dir, f"optimization_history_{study_name}.png"))

    # 2. 参数重要性图（Parameter Importances）
    fig = optuna.visualization.plot_param_importances(study)
    fig.write_image(os.path.join(save_dir, f"param_importances_{study_name}.png"))

    # 3. 超参数分布图（Hyperparameter Distributions）
    fig = optuna.visualization.plot_edf(study)
    fig.write_image(os.path.join(save_dir, f"edf_{study_name}.png"))

    # 4. 平行坐标图（Parallel Coordinates）
    fig = optuna.visualization.plot_parallel_coordinate(study)
    fig.write_image(os.path.join(save_dir, f"parallel_coordinate_{study_name}.png"))

    # 5. 片面图（Slice Plot）
    fig = optuna.visualization.plot_slice(study)
    fig.write_image(os.path.join(save_dir, f"slice_{study_name}.png"))

    # 6. 超参数交互图（Hyperparameter Interactions）
    fig = optuna.visualization.plot_contour(study)
    fig.write_image(os.path.join(save_dir, f"contour_{study_name}.png"))


# 保存每个研究的图表
save_optuna_plots(study_lit, "lit", os.path.join(base_dir, "lit"))
save_optuna_plots(study_des, "des", os.path.join(base_dir, "des"))
save_optuna_plots(study_mor, "mor", os.path.join(base_dir, "mor"))
save_optuna_plots(study_1_des, "1_des", os.path.join(base_dir, "1_des"))
save_optuna_plots(study_1_mor, "1_mor", os.path.join(base_dir, "1_mor"))
save_optuna_plots(study_mix, "mix", os.path.join(base_dir, "mix"))

print("所有贝叶斯优化的可视化图表已保存到指定目录。")