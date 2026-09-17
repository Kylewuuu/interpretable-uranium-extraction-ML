import numpy as np
import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

#%%
# 路径设置
input_path = "/home/kylewu/hotpot/examples/U_logD/data/input/"
save_dir   = "/home/kylewu/hotpot/examples/U_logD/model_para"

# 加载已保存的特征名
features_des = joblib.load(os.path.join(save_dir, "engineered_feature_names_des.pkl"))
features_mor = joblib.load(os.path.join(save_dir, "engineered_feature_names_mor.pkl"))
features_lit = joblib.load(os.path.join(save_dir, "engineered_feature_names_lit.pkl"))

# 加载原始特征数据（注意：要和训练用数据一致）
df_ligand_des = pd.read_excel(os.path.join(input_path, "ligand_des.xlsx"), engine="openpyxl")
df_sol_des    = pd.read_excel(os.path.join(input_path, "sol_des.xlsx"), engine="openpyxl")
X_des_df = pd.concat([df_ligand_des, df_sol_des], axis=1)[features_des]  # 只提取需要的特征列

df_ligand_mor = pd.read_excel(os.path.join(input_path, "ligand_mor.xlsx"), engine="openpyxl")
df_sol_mor    = pd.read_excel(os.path.join(input_path, "sol_mor.xlsx"), engine="openpyxl")
X_mor_df = pd.concat([df_ligand_mor, df_sol_mor], axis=1)[features_mor]

df_lit = pd.read_excel(os.path.join(input_path, "sol_envs.xlsx"), engine="openpyxl")
X_lit_df = df_lit[features_lit]

# 分别拟合并保存标准化器
scaler_des = StandardScaler(with_mean=False)
scaler_des.fit(X_des_df)

scaler_mor = StandardScaler(with_mean=False)
scaler_mor.fit(X_mor_df)

scaler_lit = StandardScaler(with_mean=False)
scaler_lit.fit(X_lit_df)
#%%
joblib.dump(scaler_mor, os.path.join(save_dir, "scaler_mor.pkl"))
joblib.dump(scaler_des, os.path.join(save_dir, "scaler_des.pkl"))
joblib.dump(scaler_lit, os.path.join(save_dir, "scaler_lit.pkl"))

print("✅ 所有缺失的 StandardScaler 已根据原始训练数据重新拟合并保存。")

#%%
# ---------------------- Step 1: 路径设置 ----------------------
input_ligand_mor_dir = "/home/kylewu/hotpot/examples/U_logD/pre_data/input/pub_smi_mor"
input_ligand_des_dir = "/home/kylewu/hotpot/examples/U_logD/pre_data/input/pub_smi_des"
input_smiles_dir     = "/home/kylewu/hotpot/examples/U_logD/pre_data/input/pub_smi"

sol_input_dir = "/home/kylewu/hotpot/examples/U_logD/pre_data/input"
output_dir    = "/home/kylewu/hotpot/examples/U_logD/pre_data/pred_result"
model_dir     = "/home/kylewu/hotpot/examples/U_logD/model_para"
os.makedirs(output_dir, exist_ok=True)

# ---------------------- Step 2: 加载模型与处理器 ----------------------
model = joblib.load(os.path.join(model_dir, "best_model_mix.pkl"))
scaler_des = joblib.load(os.path.join(model_dir, "scaler_des.pkl"))
scaler_mor = joblib.load(os.path.join(model_dir, "scaler_mor.pkl"))
scaler_lit = joblib.load(os.path.join(model_dir, "scaler_lit.pkl"))

features_des = joblib.load(os.path.join(model_dir, "engineered_feature_names_des.pkl"))
features_mor = joblib.load(os.path.join(model_dir, "engineered_feature_names_mor.pkl"))
features_lit = joblib.load(os.path.join(model_dir, "engineered_feature_names_lit.pkl"))
#%%
# ---------------------- Step 3: 固定溶剂部分 ----------------------
def load_and_expand(file, n_rows):
    df = pd.read_excel(os.path.join(sol_input_dir, file), engine='openpyxl')
    return pd.DataFrame({col: [df[col].iloc[0]] * n_rows for col in df.columns})

df_sol_des   = None  # 延迟加载
df_sol_mor   = None
df_sol_envs  = None

# ---------------------- Step 4: 遍历处理每一对 ligand 文件 ----------------------
for i in range(1, 6):
    tag = f"{i:02d}"
    path_des = os.path.join(input_ligand_des_dir, f"desc_part_{tag}.csv")
    path_mor = os.path.join(input_ligand_mor_dir, f"morgan_part_{tag}.csv")
    path_smi = os.path.join(input_smiles_dir, f"CID_SMILES_part_{tag}.csv")
    output_path = os.path.join(output_dir, f"pred_result_part_{tag}.csv")

    # ---- Step 4.1: 加载 ligand 特征 ----
    df_des = pd.read_csv(path_des)
    df_mor = pd.read_csv(path_mor)
    smiles = pd.read_csv(path_smi)["SMILES"]

    if len(df_des) != len(df_mor) or len(df_des) != len(smiles):
        raise ValueError(f"❌ 第 {tag} 组数据行数不一致")

    # ---- Step 4.2: 初始化溶剂特征 ----
    n_samples = len(df_des)
    if df_sol_des is None:
        df_sol_des  = load_and_expand("sol_des.xlsx", n_samples)
        df_sol_mor  = load_and_expand("sol_mor.xlsx", n_samples)
        df_sol_envs = load_and_expand("sol_envs.xlsx", n_samples)

    # ---- Step 4.3: 构造拼接后的特征 ----
    X_des = pd.concat([df_des.drop(columns=["SMILES"]), df_sol_des], axis=1)
    X_mor = pd.concat([df_mor.drop(columns=["SMILES"]), df_sol_mor], axis=1)
    X_lit = df_sol_envs

    # ---- Step 4.4: 特征选择与标准化 ----
    X_des = pd.DataFrame(scaler_des.transform(X_des[features_des]), columns=features_des)
    X_mor = pd.DataFrame(scaler_mor.transform(X_mor[features_mor]), columns=features_mor)
    X_lit = pd.DataFrame(scaler_lit.transform(X_lit[features_lit]), columns=features_lit)

    # ---- Step 4.5: 组合 & 预测 ----
    X = np.concatenate([X_des, X_mor, X_lit], axis=1)
    preds = model.predict(X)

    # ---- Step 4.6: 保存结果 ----
    df_result = pd.DataFrame({
        "SMILES": smiles,
        "logD_pred": preds
    })
    df_result.to_csv(output_path, index=False)
    print(f"✅ 已保存预测结果：{output_path}")

print("🎯 全部 50 组预测完成！")

#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -------------------- 设置路径 --------------------
pred_dir = "/home/kylewu/hotpot/examples/U_logD/pre_data/pred_result"
merged_path = "/home/kylewu/hotpot/examples/U_logD/pre_data/logD_all_predictions.csv"
top_path = "/home/kylewu/hotpot/examples/U_logD/pre_data/top_logD_SMILES.csv"

# -------------------- 合并所有预测文件 --------------------
all_parts = []
for i in range(1, 6):
    file = f"pred_result_part_{i:02d}.csv"
    df = pd.read_csv(os.path.join(pred_dir, file))
    all_parts.append(df)

full_df = pd.concat(all_parts, ignore_index=True)
full_df.to_csv(merged_path, index=False)
print(f"✅ 所有预测结果已合并保存：{merged_path}")
#%%
# -------------------- 可视化 logD 分布 --------------------
plt.figure(figsize=(8, 5))

# 用 matplotlib 画直方图
plt.hist(full_df["logD_pred"].dropna(), bins=50, density=True,
         color="skyblue", edgecolor="black", alpha=0.7, label="Histogram")

# 用 seaborn 单独叠加 KDE
sns.kdeplot(full_df["logD_pred"].dropna(), color="red", linewidth=2, label="KDE")

plt.xlabel("Predicted logD")
plt.ylabel("Density")
plt.title("Distribution of Predicted logD")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(pred_dir, "logD_distribution.png"))
plt.show()

#%%
# -------------------- 获取 logD 最高的前 N 个分子 --------------------
top_n = 1000
top_df = full_df.sort_values(by="logD_pred", ascending=False).head(top_n)
top_df.reset_index(drop=True, inplace=True)
top_df.to_csv(top_path, index=False)
print(f"✅ Top {top_n} 高 logD 分子的 SMILES 和值已保存：{top_path}")

'''
#%%
from sklearn.neighbors import NearestNeighbors
from scipy.stats import gaussian_kde
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ---------------- Step 1: 构造训练集特征矩阵 ----------------
X_train_des = scaler_des.transform(
    pd.concat([df_ligand_des, df_sol_des], axis=1)[features_des]
)
X_train_mor = scaler_mor.transform(
    pd.concat([df_ligand_mor, df_sol_mor], axis=1)[features_mor]
)

X_train = np.concatenate([X_train_des, X_train_mor], axis=1)
print(f"训练集特征矩阵大小: {X_train.shape}")

#%%
# ---------------- Step 2: 构造 Top1000 配体特征（填充溶剂列为 0） ----------------
X_top_list = []
for i in range(1, 6):
    tag = f"{i:02d}"
    path_des = os.path.join(input_ligand_des_dir, f"desc_part_{tag}.csv")
    path_mor = os.path.join(input_ligand_mor_dir, f"morgan_part_{tag}.csv")
    path_smi = os.path.join(input_smiles_dir, f"CID_SMILES_part_{tag}.csv")

    df_des = pd.read_csv(path_des).drop(columns=["SMILES"])
    df_mor = pd.read_csv(path_mor).drop(columns=["SMILES"])
    df_smi = pd.read_csv(path_smi)["SMILES"]

    df_top_merge = pd.DataFrame({"SMILES": df_smi})
    df_top_merge = df_top_merge.merge(top_df[["SMILES"]], on="SMILES")

    if not df_top_merge.empty:
        idx = df_top_merge.index

        # -------- des 部分：补齐缺失列为 0 --------
        missing_des_cols = [f for f in features_des if f not in df_des.columns]
        df_des_full = pd.concat(
            [df_des, pd.DataFrame(0, index=df_des.index, columns=missing_des_cols)], axis=1
        )
        X_des = scaler_des.transform(df_des_full.iloc[idx][features_des])

        # -------- mor 部分：补齐缺失列为 0 --------
        missing_mor_cols = [f for f in features_mor if f not in df_mor.columns]
        df_mor_full = pd.concat(
            [df_mor, pd.DataFrame(0, index=df_mor.index, columns=missing_mor_cols)], axis=1
        )
        X_mor = scaler_mor.transform(df_mor_full.iloc[idx][features_mor])

        # -------- 拼接（只 des+mor，无 lit） --------
        X_concat = np.concatenate([X_des, X_mor], axis=1)
        X_top_list.append(X_concat)

X_top = np.vstack(X_top_list)
print(f"Top1000 配体特征矩阵大小: {X_top.shape}")

#%%
# ---------------- Step 2.5: 缺失值填充 ----------------
# 将 NaN 统一填充为 0（保证 NearestNeighbors 可用）
X_train = np.nan_to_num(X_train, nan=0.0)
X_top   = np.nan_to_num(X_top, nan=0.0)

print(f"训练集 NaN 数量: {np.isnan(X_train).sum()}")
print(f"Top1000 NaN 数量: {np.isnan(X_top).sum()}")

#%%
# ---------------- Step 3: 最近邻分析 ----------------
nbrs_train = NearestNeighbors(n_neighbors=2).fit(X_train)
distances_train, _ = nbrs_train.kneighbors(X_train)
train_nn_distances = distances_train[:,1]

mean_train = np.mean(train_nn_distances)
q95_train = np.percentile(train_nn_distances, 95)
print(f"训练集最近邻平均距离: {mean_train:.2f}")
print(f"训练集最近邻95%分位数: {q95_train:.2f}")

nbrs = NearestNeighbors(n_neighbors=1).fit(X_train)
distances, _ = nbrs.kneighbors(X_top)

mean_top = np.mean(distances)
max_top  = np.max(distances)
print(f"Top1000 最近邻平均距离: {mean_top:.2f}")
print(f"Top1000 最近邻最大距离: {max_top:.2f}")

# ---------------- Step 4: 可视化 ----------------
plt.figure(figsize=(8,5))
plt.hist(train_nn_distances, bins=30, density=True, color="green", alpha=0.5, label="Training Ligands NN")
plt.hist(distances.flatten(), bins=30, density=True, color="skyblue", alpha=0.6, label="Top1000 → Training")
kde = gaussian_kde(distances.flatten())
x_vals = np.linspace(min(distances.flatten()), max(distances.flatten()), 200)
plt.plot(x_vals, kde(x_vals), color="red", linewidth=2, label="Top1000 KDE")
plt.axvline(q95_train, color="red", linestyle="--", label=f"95% Train Threshold = {q95_train:.2f}")
plt.axvline(mean_top, color="blue", linestyle="--", label=f"Top1000 Mean = {mean_top:.2f}")
plt.xlabel("Nearest Neighbor Distance (Ligand Features Only)")
plt.ylabel("Density")
plt.title("NN Distance Distribution: Training vs Top1000 (Ligand Features)")
plt.legend()
plt.grid(True)
plt.show()

# ---------------- Step 5: 标记 OOD 分子（多阈值策略） ----------------

# 阈值1：95% 分位数
threshold_q95 = q95_train
ood_mask_q95 = distances.flatten() > threshold_q95
ood_count_q95 = np.sum(ood_mask_q95)
print(f"[95%分位数阈值] Top1000 中疑似 OOD 分子数: {ood_count_q95}/{len(X_top)}")

# 阈值2：均值 + 3σ
threshold_mean_std = mean_train + 3 * np.std(train_nn_distances)
ood_mask_meanstd = distances.flatten() > threshold_mean_std
ood_count_meanstd = np.sum(ood_mask_meanstd)
print(f"[均值+3σ阈值] Top1000 中疑似 OOD 分子数: {ood_count_meanstd}/{len(X_top)}")

# 可视化：训练集 vs Top1000 最近邻距离分布 + 两条阈值线
plt.figure(figsize=(8,5))
plt.hist(train_nn_distances, bins=30, density=True, color="green", alpha=0.5, label="Training Ligands NN")
plt.hist(distances.flatten(), bins=30, density=True, color="skyblue", alpha=0.6, label="Top1000 → Training")

# KDE 平滑曲线
from scipy.stats import gaussian_kde
kde = gaussian_kde(distances.flatten())
x_vals = np.linspace(min(distances.flatten()), max(distances.flatten()), 200)
plt.plot(x_vals, kde(x_vals), color="red", linewidth=2, label="Top1000 KDE")

# 阈值辅助线
plt.axvline(threshold_q95, color="red", linestyle="--", label=f"95% Train Threshold = {threshold_q95:.2f}")
plt.axvline(threshold_mean_std, color="purple", linestyle="--", label=f"Mean+3σ = {threshold_mean_std:.2f}")
plt.axvline(mean_top, color="blue", linestyle="--", label=f"Top1000 Mean = {mean_top:.2f}")

plt.xlabel("Nearest Neighbor Distance (Ligand Features Only)")
plt.ylabel("Density")
plt.title("NN Distance Distribution with Multiple Thresholds")
plt.legend()
plt.grid(True)
plt.show()

# 导出 OOD 分子（可选）
ood_smis_q95 = [top_smis[i] for i, flag in enumerate(ood_mask_q95) if flag]
ood_smis_meanstd = [top_smis[i] for i, flag in enumerate(ood_mask_meanstd) if flag]

ood_df_q95 = pd.DataFrame({"SMILES": ood_smis_q95, "Threshold": "95% Quantile"})
ood_df_meanstd = pd.DataFrame({"SMILES": ood_smis_meanstd, "Threshold": "Mean+3σ"})
ood_df = pd.concat([ood_df_q95, ood_df_meanstd], ignore_index=True)

print("前几个疑似 OOD 分子：")
print(ood_df.head())
'''
