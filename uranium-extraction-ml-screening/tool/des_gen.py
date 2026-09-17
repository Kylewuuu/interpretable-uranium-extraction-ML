import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Descriptors import CalcMolDescriptors

# 更新后的输入输出目录
input_dir = '/home/kylewu/hotpot/examples/U_logD/pre_data/input/pub_smi'
output_dir = '/home/kylewu/hotpot/examples/U_logD/pre_data/input/pub_smi_des'
os.makedirs(output_dir, exist_ok=True)

# 指定 3D 描述符名称
attr3d = [
    'InertialShapeFactor', 'NPR1', 'NPR2', 'Asphericity', 'Eccentricity',
    'PMI1', 'PMI2', 'PMI3', 'RadiusOfGyration', 'SpherocityIndex'
]

# 主函数：提取分子描述符
def get_rdkit_mol_descriptor(list_smi):
    descriptors_2d = []
    descriptors_3d = []

    for idx, smi in enumerate(list_smi):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                raise ValueError("RDKit failed to parse SMILES.")

            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)

            # 2D
            desc2d = CalcMolDescriptors(mol)

            # 3D
            desc3d = {}
            for n in attr3d:
                if hasattr(Chem.Descriptors3D, n):
                    try:
                        desc3d[n] = getattr(Chem.Descriptors3D, n)(mol)
                    except Exception:
                        desc3d[n] = None

            descriptors_2d.append(desc2d)
            descriptors_3d.append(desc3d)

        except Exception as e:
            print(f"[{idx}] Error: {smi} -> {e}")
            descriptors_2d.append({name: None for name, _ in Descriptors._descList})
            descriptors_3d.append({key: None for key in attr3d})

    df_2d = pd.DataFrame(descriptors_2d)
    df_3d = pd.DataFrame(descriptors_3d)
    return pd.concat([df_2d, df_3d], axis=1)

# 遍历处理 50 个 SMILES 文件
for i in range(1, 2):
    file_name = f'CID_SMILES_part_{i:02d}.csv'
    input_path = os.path.join(input_dir, file_name)
    output_path = os.path.join(output_dir, f'desc_part_{i:02d}.csv')

    # 读取 SMILES 列
    df = pd.read_csv(input_path)
    smiles_list = df['SMILES'].tolist()

    # 获取描述符
    desc_df = get_rdkit_mol_descriptor(smiles_list)

    # 合并 SMILES 列
    result_df = pd.concat([df[['SMILES']], desc_df], axis=1)

    # 添加前缀 lig_ 到所有描述符列
    result_df.rename(columns={col: f'lig_{col}' for col in result_df.columns if col != 'SMILES'}, inplace=True)

    # 保存到输出文件
    result_df.to_csv(output_path, index=False)
    print(f'保存文件：{output_path}（{len(result_df)} 行）')

print('✅ 所有分子描述符提取并保存完成。')
