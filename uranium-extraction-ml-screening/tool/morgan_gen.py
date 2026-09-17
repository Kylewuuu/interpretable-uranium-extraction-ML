import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

input_dir = '/home/kylewu/hotpot/examples/U_logD/pre_data/input/pub_smi'
output_dir = '/home/kylewu/hotpot/examples/U_logD/pre_data/input/pub_smi_mor'
os.makedirs(output_dir, exist_ok=True)

def extract_morgan_fingerprint(smiles):
    if pd.isna(smiles):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
        return list(fp)
    return None

chunksize = 100000

for i in range(1, 51):
    tag = f'{i:02d}'
    input_path = os.path.join(input_dir, f'CID_SMILES_part_{tag}.csv')
    output_path = os.path.join(output_dir, f'morgan_part_{tag}.csv')

    if os.path.exists(output_path):
        print(f'✅ 已存在：{output_path}，跳过处理')
        continue

    print(f'🚀 开始处理：{input_path}')
    is_first_chunk = True

    for chunk in pd.read_csv(input_path, chunksize=chunksize):
        chunk['Morgan_Fingerprint'] = chunk['SMILES'].apply(extract_morgan_fingerprint)
        fp_df = chunk['Morgan_Fingerprint'].apply(pd.Series)
        fp_df.columns = [f'lig_{j}' for j in range(fp_df.shape[1])]
        result_df = pd.concat([chunk['SMILES'], fp_df], axis=1)

        result_df.to_csv(output_path, mode='w' if is_first_chunk else 'a',
                         header=is_first_chunk, index=False)
        is_first_chunk = False

    print(f'✅ 已完成并保存：{output_path}')

print('🎉 所有未完成的文件已补全完成。')
