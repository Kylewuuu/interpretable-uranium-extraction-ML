"""import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import shap
import matplotlib.pyplot as plt

def main():
    # 定义输入文件路径
    input_dir = '/home/kylewu/hotpot/examples/U_logD/data/input'
    lit_path = f'{input_dir}/sol_envs.xlsx'      # 文献环境数据
    ligand_des_path = f'{input_dir}/ligand_des.xlsx'  # 配体描述符
    sol_des_path = f'{input_dir}/sol_des.xlsx'        # 溶剂描述符
    ligand_mor_path = f'{input_dir}/ligand_mor.xlsx'  # 配体Morgan指纹
    sol_mor_path = f'{input_dir}/sol_mor.xlsx'        # 溶剂Morgan指纹
    target_path = f'{input_dir}/y.xlsx'               # 目标值 (logD)

    # 定义模型参数和输出文件路径
    model_dir = '/home/kylewu/hotpot/examples/U_logD/model_para'
    features_path = f'{model_dir}/engineered_feature_names_mix.pkl'  # 融合特征名列表
    model_path = f'{model_dir}/best_model_mix.pkl'                   # 融合模型文件

    # 定义保存SHAP图像的目录和文件名
    shap_dir = '/home/kylewu/hotpot/examples/U_logD/shap'
    output_plot1 = f'{shap_dir}/Fig3a_SHAP_summary.pdf'      # SHAP分析图
    output_plot2 = f'{shap_dir}/Fig3b_SHAP_c_HNO3.pdf'       # SHAP依赖图: c_HNO3
    output_plot3 = f'{shap_dir}/Fig3c_SHAP_lig_VSA_EState5.pdf'  # SHAP依赖图: lig_VSA_EState5

    # 1. 读取数据
    df_lit = pd.read_excel(lit_path, engine='openpyxl')
    df_ligand_des = pd.read_excel(ligand_des_path, engine='openpyxl')
    df_sol_des = pd.read_excel(sol_des_path, engine='openpyxl')
    df_ligand_mor = pd.read_excel(ligand_mor_path, engine='openpyxl')
    df_sol_mor = pd.read_excel(sol_mor_path, engine='openpyxl')
    df_y = pd.read_excel(target_path, engine='openpyxl')

    # 2. 数据预处理：缺失值填充（列均值）和标准化
    df_lit_filled = df_lit.fillna(df_lit.mean())
    scaler_lit = StandardScaler(with_mean=False)
    lit_scaled = scaler_lit.fit_transform(df_lit_filled)
    df_lit_scaled = pd.DataFrame(lit_scaled, columns=df_lit_filled.columns, index=df_lit_filled.index)

    df_desc = pd.concat([df_ligand_des, df_sol_des], axis=1)
    df_desc_filled = df_desc.fillna(df_desc.mean())
    scaler_desc = StandardScaler(with_mean=False)
    desc_scaled = scaler_desc.fit_transform(df_desc_filled)
    df_desc_scaled = pd.DataFrame(desc_scaled, columns=df_desc_filled.columns, index=df_desc_filled.index)

    df_mor = pd.concat([df_ligand_mor, df_sol_mor], axis=1)
    df_mor_filled = df_mor.fillna(df_mor.mean())
    scaler_mor = StandardScaler(with_mean=False)
    mor_scaled = scaler_mor.fit_transform(df_mor_filled)
    df_mor_scaled = pd.DataFrame(mor_scaled, columns=df_mor_filled.columns, index=df_mor_filled.index)

    # 3. 合并特征集: descriptor + morgan + literature (假设样本行顺序一致)
    X_fused = pd.concat([df_desc_scaled, df_mor_scaled, df_lit_scaled], axis=1)

    # 4. 读取融合模型使用的特征名列表
    with open(features_path, 'rb') as f:
        feature_names = pickle.load(f)

    # 5. 筛选融合特征集中所需的特征列，构建 X_mix 特征矩阵
    X_mix = X_fused.loc[:, feature_names]

    # 6. 加载融合模型并计算SHAP值
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_mix)

    # 7. 绘制SHAP分析图并保存（与原代码B相同的步骤和配色方案）
    shap.summary_plot(shap_values, X_mix, feature_names=X_mix.columns, plot_type="dot",
                      color_bar=True, cmap='RdBu', show=False)
    plt.savefig(output_plot1, format='pdf', dpi=300)
    plt.clf()

    # 8. 绘制SHAP依赖图并保存（与原代码B保持一致的配色方案）
    shap.dependence_plot('c_HNO3', shap_values, X_mix, show=False, cmap='RdBu')
    plt.savefig(output_plot2, format='pdf', dpi=300)
    plt.clf()
    shap.dependence_plot('lig_VSA_EState5', shap_values, X_mix, show=False, cmap='RdBu')
    plt.savefig(output_plot3, format='pdf', dpi=300)
    plt.clf()

    print(f'SHAP分析图已保存为 {output_plot1}，SHAP依赖图已保存为 {output_plot2} 和 {output_plot3}')

if __name__ == "__main__":
    main()
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import shap
import matplotlib.pyplot as plt
import os

def main():
    # 定义输入文件路径
    input_dir = '/home/kylewu/hotpot/examples/U_logD/data/input'
    lit_path = f'{input_dir}/sol_envs.xlsx'      # 文献环境数据
    ligand_des_path = f'{input_dir}/ligand_des.xlsx'  # 配体描述符
    sol_des_path = f'{input_dir}/sol_des.xlsx'        # 溶剂描述符
    ligand_mor_path = f'{input_dir}/ligand_mor.xlsx'  # 配体Morgan指纹
    sol_mor_path = f'{input_dir}/sol_mor.xlsx'        # 溶剂Morgan指纹
    target_path = f'{input_dir}/y.xlsx'               # 目标值 (logD)

    # 定义模型参数和输出文件路径
    model_dir = '/home/kylewu/hotpot/examples/U_logD/model_para'
    features_path = f'{model_dir}/engineered_feature_names_mix.pkl'  # 融合特征名列表
    model_path = f'{model_dir}/best_model_mix.pkl'                   # 融合模型文件
    # 输出SHAP图片目录
    shap_dir = '/home/kylewu/hotpot/examples/U_logD/shap'
    os.makedirs(shap_dir, exist_ok=True)

    # 1. 读取数据
    df_lit = pd.read_excel(lit_path, engine='openpyxl')
    df_ligand_des = pd.read_excel(ligand_des_path, engine='openpyxl')
    df_sol_des = pd.read_excel(sol_des_path, engine='openpyxl')
    df_ligand_mor = pd.read_excel(ligand_mor_path, engine='openpyxl')
    df_sol_mor = pd.read_excel(sol_mor_path, engine='openpyxl')
    df_y = pd.read_excel(target_path, engine='openpyxl')

    # 2. 数据预处理：缺失值填充（列均值）和标准化（StandardScaler, 不去中心化）
    df_lit_filled = df_lit.fillna(df_lit.mean())
    scaler_lit = StandardScaler(with_mean=False)
    lit_scaled = scaler_lit.fit_transform(df_lit_filled)
    df_lit_scaled = pd.DataFrame(lit_scaled, columns=df_lit_filled.columns, index=df_lit_filled.index)

    df_desc = pd.concat([df_ligand_des, df_sol_des], axis=1)
    df_desc_filled = df_desc.fillna(df_desc.mean())
    scaler_desc = StandardScaler(with_mean=False)
    desc_scaled = scaler_desc.fit_transform(df_desc_filled)
    df_desc_scaled = pd.DataFrame(desc_scaled, columns=df_desc_filled.columns, index=df_desc_filled.index)

    df_mor = pd.concat([df_ligand_mor, df_sol_mor], axis=1)
    df_mor_filled = df_mor.fillna(df_mor.mean())
    scaler_mor = StandardScaler(with_mean=False)
    mor_scaled = scaler_mor.fit_transform(df_mor_filled)
    df_mor_scaled = pd.DataFrame(mor_scaled, columns=df_mor_filled.columns, index=df_mor_filled.index)

    # 3. 合并特征集: descriptor + morgan + literature
    X_fused = pd.concat([df_desc_scaled, df_mor_scaled, df_lit_scaled], axis=1)

    # 4. 读取融合模型使用的特征名列表
    with open(features_path, 'rb') as f:
        feature_names = pickle.load(f)

    # 5. 筛选融合特征集中所需的特征列，构建 X_mix 特征矩阵
    X_mix = X_fused.loc[:, feature_names]

    # 6. 加载融合模型并计算SHAP值
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_mix)

    # 7. 绘制SHAP总结图：蜂群图和条形图合并，并保存
    plt.figure(figsize=(15, 10), dpi=300)
    # 绘制蜂群散点图并调整位置
    shap.summary_plot(shap_values, X_mix, feature_names=feature_names, plot_type="dot",
                      cmap='RdBu', show=False, color_bar=True)
    plt.gca().set_position([0.2, 0.2, 0.65, 0.65])  # 留出空间
    ax1 = plt.gca()
    # 在顶部绘制条形图（平均SHAP值）
    ax2 = ax1.twiny()
    shap.summary_plot(shap_values, X_mix, feature_names=feature_names, plot_type="bar", show=False)
    plt.gca().set_position([0.2, 0.2, 0.65, 0.65])
    # 添加横线分隔（y取一半的特征数量）
    half = len(feature_names) // 2
    ax2.axhline(y=half, color='k', linestyle='-', linewidth=1.5)
    # 调整条形图透明度
    for bar in ax2.patches:
        bar.set_facecolor('gray')
        bar.set_alpha(0.4)
    # 设置坐标轴标签
    ax1.set_xlabel('Shapley Value Contribution', fontsize=10)
    ax2.set_xlabel('Mean Shapley Value', fontsize=10)
    ax2.xaxis.set_label_position('top')
    ax2.xaxis.tick_top()
    ax1.set_ylabel('Features', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{shap_dir}/shap_summary.png', format='png', dpi=300)
    plt.clf()

    # 8. 绘制SHAP依赖图并保存
    # 特征 c_HNO3 的SHAP依赖图
    shap.dependence_plot('c_HNO3', shap_values, X_mix, show=False, cmap='RdBu')
    plt.savefig(f'{shap_dir}/c_HNO3.png', format='png', dpi=300)
    plt.clf()

    # 特征 lig_VSA_EState5 的SHAP依赖图
    shap.dependence_plot('c_lig', shap_values, X_mix, show=False, cmap='RdBu')
    plt.savefig(f'{shap_dir}/c_lig.png', format='png', dpi=300)
    plt.clf()

    # 特征 lig_VSA_EState5 的SHAP依赖图
    shap.dependence_plot('lig_VSA_EState5', shap_values, X_mix, show=False, cmap='RdBu')
    plt.savefig(f'{shap_dir}/lig_VSA_EState5.png', format='png', dpi=300)
    plt.clf()

    # 特征 lig_VSA_EState5 的SHAP依赖图
    shap.dependence_plot('c_Metal_mM', shap_values, X_mix, show=False, cmap='RdBu')
    plt.savefig(f'{shap_dir}/c_Metal_mM.png', format='png', dpi=300)
    plt.clf()

    # 特征 lig_VSA_EState5 的SHAP依赖图
    shap.dependence_plot('lig_AvgIpc', shap_values, X_mix, show=False, cmap='RdBu')
    plt.savefig(f'{shap_dir}/lig_AvgIpc.png', format='png', dpi=300)
    plt.clf()

    # 特征 lig_VSA_EState5 的SHAP依赖图
    shap.dependence_plot('lig_619', shap_values, X_mix, show=False, cmap='RdBu')
    plt.savefig(f'{shap_dir}/lig_619.png', format='png', dpi=300)
    plt.clf()

    # 特征 lig_VSA_EState5 的SHAP依赖图
    shap.dependence_plot('lig_1017', shap_values, X_mix, show=False, cmap='RdBu')
    plt.savefig(f'{shap_dir}/lig_1017.png', format='png', dpi=300)
    plt.clf()

    # 特征 lig_VSA_EState5 的SHAP依赖图
    shap.dependence_plot('lig_348', shap_values, X_mix, show=False, cmap='RdBu')
    plt.savefig(f'{shap_dir}/lig_348.png', format='png', dpi=300)
    plt.clf()

    # 特征 lig_VSA_EState5 的SHAP依赖图
    shap.dependence_plot('sol_849', shap_values, X_mix, show=False, cmap='RdBu')
    plt.savefig(f'{shap_dir}/sol_849.png', format='png', dpi=300)
    plt.clf()

    # 特征 lig_VSA_EState5 的SHAP依赖图
    shap.dependence_plot('lig_978', shap_values, X_mix, show=False, cmap='RdBu')
    plt.savefig(f'{shap_dir}/lig_978.png', format='png', dpi=300)
    plt.clf()

    # 特征 lig_VSA_EState5 的SHAP依赖图
    shap.dependence_plot('sol_896', shap_values, X_mix, show=False, cmap='RdBu')
    plt.savefig(f'{shap_dir}/sol_896.png', format='png', dpi=300)
    plt.clf()



    print(f'SHAP图像已保存至 {shap_dir} 目录下。')

if __name__ == "__main__":
    main()
