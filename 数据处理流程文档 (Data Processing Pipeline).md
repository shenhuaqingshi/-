# 数据处理流程文档 (Data Processing Pipeline)

## 概述

本文档旨在详细记录从原始数据（如晶体结构文件、性能数据）到可用于机器学习模型训练的标准化数据集的完整处理流程。本流程确保了整个过程的透明度和可复现性。

**输入:**

* `benchmark_set.json`: 包含已知材料性能范围的基准数据。
* `cif_files/` 目录: 包含所有目标材料的 `.cif` 晶体结构文件。

**输出:**

* `processed_features.csv`: 一个经过清洗、特征提取和标准化后的表格化数据集，每一行代表一个材料样本，每一列代表一个特征或标签。

---

## 1. 环境准备与依赖项

首先，安装必要的Python库。

```bash
pip install pandas pymatgen scikit-learn numpy
```

---

## 2. 代码实现

### 2.1 导入库

```python
import os
import json
import pandas as pd
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from sklearn.preprocessing import StandardScaler
import numpy as np
```

### 2.2 步骤一：加载基准数据 (`benchmark_set.json`)

此步骤读取JSON文件，将其解析为Python字典，并提取关键信息以备后续使用。

**规则:**

* 加载JSON文件。
* 遍历`benchmark_materials`列表，提取`material_id`、`name`和`properties`。
* 将`properties`中的范围（如`[130, 160]`）转换为中间值，作为该材料的代表性性能。

```python
def load_benchmark_data(json_path):
    """
    Loads benchmark data from a JSON file.

    Args:
        json_path (str): Path to the benchmark_set.json file.

    Returns:
        dict: A dictionary mapping material IDs to their properties.
    """
    with open(json_path, 'r') as f:
        raw_data = json.load(f)
    
    benchmark_dict = {}
    for material in raw_data['benchmark_materials']:
        mat_id = material['material_id']
        # For simplicity, we take the midpoint of ranges as the value
        processed_props = {}
        for key, val in material['properties'].items():
            if isinstance(val, list) and len(val) == 2:
                # Calculate midpoint for range
                processed_props[key] = sum(val) / 2.0
            else:
                processed_props[key] = val
        
        benchmark_dict[mat_id] = {
            'name': material['name'],
            'properties': processed_props
        }
    return benchmark_dict

# --- Execution ---
benchmark_data = load_benchmark_data('benchmark_set.json')
print("Loaded benchmark data for materials:", list(benchmark_data.keys()))
```

### 2.3 步骤二：从CIF文件中提取晶体学特征

此步骤遍历`cif_files`目录，为每个CIF文件计算一组基础的晶体学特征。

**规则:**

* 遍历指定目录下的所有`.cif`文件。
* 使用`pymatgen`库读取CIF文件，创建`Structure`对象。
* 从`Structure`对象中提取以下特征：
  * `volume_per_atom`: 单个原子占据的体积。
  * `density`: 晶体密度。
  * `spacegroup_number`: 空间群国际号。
  * `crystal_system`: 晶系（通过空间群分析获得）。
  * `formula`: 简化的化学式。

```python
def extract_structure_features(cif_dir):
    """
    Extracts structural features from all CIF files in a directory.

    Args:
        cif_dir (str): Path to the directory containing CIF files.

    Returns:
        dict: A dictionary mapping material IDs (extracted from filename) to their features.
    """
    features = {}
    for filename in os.listdir(cif_dir):
        if filename.endswith('.cif'):
            material_id = filename.split('.')[0].split('__')[0]
            cif_path = os.path.join(cif_dir, filename)
            
            try:
                structure = Structure.from_file(cif_path)
                
                # Calculate features
                analyzer = SpacegroupAnalyzer(structure)
                volume_per_atom = structure.volume / len(structure)
                density = structure.density
                spacegroup_num = analyzer.get_space_group_number()
                crystal_system = analyzer.get_crystal_system()

                features[material_id] = {
                    'volume_per_atom': volume_per_atom,
                    'density': density,
                    'spacegroup_number': spacegroup_num,
                    'crystal_system': crystal_system,
                    'formula': str(structure.composition.reduced_formula)
                }
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    return features

# --- Execution ---
structure_features = extract_structure_features('cif_files/')
print("Extracted structure features for materials:", list(structure_features.keys()))
```

### 2.4 步骤三：合并数据并创建主数据框

此步骤将从基准文件和CIF文件中提取的数据合并到一个统一的Pandas DataFrame中。

**规则:**

* 创建一个空的DataFrame。
* 遍历基准数据中的所有材料。如果该材料也存在于结构特征字典中，则将其基准性能属性和结构特征合并为一行。
* 忽略任何在任一数据源中缺失的材料，以确保数据完整性。

```python
# --- Execution ---
# Initialize an empty list to store rows
all_data_rows = []

for mat_id, bench_info in benchmark_data.items():
    if mat_id in structure_features:
        row_data = {'material_id': mat_id}
        
        # Add structure features
        row_data.update(structure_features[mat_id])
        
        # Add performance properties from benchmark
        row_data.update(bench_info['properties'])
        
        all_data_rows.append(row_data)
    else:
        print(f"Warning: Material {mat_id} has benchmark data but no CIF file.")

# Create the final DataFrame
df = pd.DataFrame(all_data_rows)
print("\nFinal DataFrame shape:", df.shape)
print(df.head())
```

### 2.5 步骤四：数据清洗

此步骤检查并处理数据中的潜在问题，例如缺失值或异常值。

**规则:**

* 检查是否存在任何NaN值。
* 如果存在NaN值，可以选择删除含有NaN的整行（`dropna`），或使用其他策略填充。在此示例中，我们将删除它们。
* 检查并打印数据的基本统计信息，以发现潜在的异常值。

```python
# --- Execution ---
print("\n--- Data Cleaning ---")
print("Shape before cleaning:", df.shape)
print("Missing values:\n", df.isnull().sum())

# Drop any rows with missing values
df_clean = df.dropna()
print("Shape after dropping NaNs:", df_clean.shape)

# Display basic statistics for numerical columns
print("\nBasic statistics:")
print(df_clean.describe())
```

### 2.6 步骤五：特征工程与标准化

此步骤准备最终的特征矩阵(X)和目标向量(y)，并对特征进行标准化处理，使其均值为0，方差为1。

**规则:**

* 确定哪些列是特征，哪些是目标变量。例如，`practical_capacity_mAh_g`是我们可能预测的目标。
* 将特征列和目标列分离。
* 使用`sklearn`的`StandardScaler`对特征进行标准化。

```python
# --- Execution ---
print("\n--- Feature Engineering & Standardization ---")

# Define features (X) and target (y)
feature_columns = ['volume_per_atom', 'density', 'spacegroup_number'] # Example features
target_column = 'practical_capacity_mAh_g' # Example target

X = df_clean[feature_columns]
y = df_clean[target_column]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create a new DataFrame for standardized features
X_scaled_df = pd.DataFrame(X_scaled, columns=[f"{col}_scaled" for col in feature_columns], index=df_clean.index)

# Combine standardized features with the target
final_df = pd.concat([X_scaled_df, y], axis=1)

print("Final processed DataFrame (first 5 rows):")
print(final_df)
```

### 2.7 步骤六：保存最终数据集

最后一步是将处理完成的DataFrame保存为CSV文件，以便后续建模使用。

```python
# --- Execution ---
output_filename = 'processed_features.csv'
final_df.to_csv(output_filename, index=False)
print(f"\nProcessed dataset saved to '{output_filename}'")
```

---

## 3. 总结

本流程文档描述了一个端到端的数据处理管道。它从结构化的JSON和CIF文件开始，通过加载、特征提取、合并、清洗和标准化等步骤，最终生成一个可用于机器学习的干净、标准化的数据集(`processed_features.csv`)。通过遵循此文档中的规则和代码，可以确保数据处理过程的可复现性。
