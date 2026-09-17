import gzip
import csv
import pandas as pd
from rdkit import Chem

def count_smiles(file_path):
    with open(file_path, 'r') as f:
        next(f)  # 跳过表头
        return sum(1 for _ in f)

#%%
file_path = '/home/kylewu/hotpot/examples/CID-SMILES.gz'
output_path = './CID_SMILES.csv'

with gzip.open(file_path, 'rt') as f_in, open(output_path, 'w', newline='') as f_out:
    writer = csv.writer(f_out)
    writer.writerow(['CID', 'SMILES'])  # 写入表头

    for line in f_in:
        cid, smiles = line.strip().split('\t')
        writer.writerow([cid, smiles])

print(f'已成功保存到 {output_path}')
#%%

input_path = '/home/kylewu/hotpot/examples/CID_SMILES.csv'
output_path = '/home/kylewu/hotpot/examples/mol.csv'

amide_query = Chem.MolFromSmarts('OC1=C(C)[C@@](C[C@](O2)([H])[C@]34[C@]5([H])[C@@H](O)[C@H](O)[C@](OC4)(C(OC)=O)[C@]3([H])[C@@H](OC(/C=C(C)\C)=O)C2=O)([H])[C@]5(C)CC1=O')

chunksize = 100000  # 每批读取10万行，可根据内存调整

# 写入表头
with open(output_path, 'w') as f_out:
    f_out.write('CID,SMILES\n')

# 分块读取 + 筛选 + 追加写入
for chunk in pd.read_csv(input_path, chunksize=chunksize):
    filtered_rows = []
    for idx, row in chunk.iterrows():
        smi = row['SMILES']
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        if mol.HasSubstructMatch(amide_query):
            filtered_rows.append({'CID': row['CID'], 'SMILES': smi})
    if filtered_rows:
        pd.DataFrame(filtered_rows).to_csv(output_path, mode='a', index=False, header=False)

print(f'筛选完成，结果已写入 {output_path}')

#%%
print('原始文件分子数:', count_smiles('/home/kylewu/hotpot/examples/CID_SMILES.csv'))
print('筛选后分子数:', count_smiles('/home/kylewu/hotpot/examples/mol.csv'))

#%% 进一步筛选：原子数 < 100
input_path2 = '/home/kylewu/hotpot/examples/CID_SMILES_filtered_amide.csv'
output_path2 = '/home/kylewu/hotpot/examples/CID_SMILES_filtered_amide_atomcount.csv'

chunksize = 500000

with open(output_path2, 'w') as f_out:
    f_out.write('CID,SMILES\n')

for chunk in pd.read_csv(input_path2, chunksize=chunksize):
    filtered_rows = []
    for idx, row in chunk.iterrows():
        smi = row['SMILES']
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        if mol.GetNumAtoms() < 100:
            filtered_rows.append({'CID': row['CID'], 'SMILES': smi})
    if filtered_rows:
        pd.DataFrame(filtered_rows).to_csv(output_path2, mode='a', index=False, header=False)

print(f'进一步筛选完成，结果已写入 {output_path2}')
print('最终筛选后分子数:', count_smiles(output_path2))

#%% 第三步筛选：不含金属元素
input_path3 = '/home/kylewu/hotpot/examples/CID_SMILES_filtered_amide_atomcount.csv'
output_path3 = '/home/kylewu/hotpot/examples/CID_SMILES_filtered_amide_atomcount_nonmetal.csv'

# 常见金属元素原子序号，参考周期表
metal_atomic_numbers = [
    3, 4, 11, 12, 13, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    31, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 55, 56,
    57, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90,
    91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103
]

chunksize = 500000

with open(output_path3, 'w') as f_out:
    f_out.write('CID,SMILES\n')

for chunk in pd.read_csv(input_path3, chunksize=chunksize):
    filtered_rows = []
    for idx, row in chunk.iterrows():
        smi = row['SMILES']
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        if any(atom.GetAtomicNum() in metal_atomic_numbers for atom in mol.GetAtoms()):
            continue  # 含有金属，跳过
        filtered_rows.append({'CID': row['CID'], 'SMILES': smi})
    if filtered_rows:
        pd.DataFrame(filtered_rows).to_csv(output_path3, mode='a', index=False, header=False)

print(f'最终筛选完成，结果已写入 {output_path3}')
print('最终筛选后分子数:', count_smiles(output_path3))
#%%
import pandas as pd
import os

# 输入路径和输出目录
input_path = '/home/kylewu/hotpot/examples/CID_SMILES_filtered_amide_atomcount_nonmetal.csv'
output_dir = '/home/kylewu/hotpot/examples/U_logD/pre_data/input/pub_smi'
os.makedirs(output_dir, exist_ok=True)

# 读取数据
df = pd.read_csv(input_path)

# 拆分为 50 份
num_parts = 50
total_rows = len(df)
rows_per_file = total_rows // num_parts
remainder = total_rows % num_parts

start_idx = 0
for i in range(num_parts):
    end_idx = start_idx + rows_per_file + (1 if i < remainder else 0)
    part_df = df.iloc[start_idx:end_idx]

    # 生成文件名并保存
    file_name = f'CID_SMILES_part_{i+1:02d}.csv'
    file_path = os.path.join(output_dir, file_name)
    part_df.to_csv(file_path, index=False)

    print(f'保存：{file_path}，包含 {len(part_df)} 行')

    start_idx = end_idx

print(f'总共生成 {num_parts} 个文件，保存于目录：{output_dir}')
