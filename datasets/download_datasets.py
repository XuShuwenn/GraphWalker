import os
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import json

# 可选：为 HF Hub 设置更稳健的超时（可被环境变量覆盖）
os.environ.setdefault("HF_HUB_READ_TIMEOUT", "60")  # 秒
os.environ.setdefault("HF_HUB_CONNECT_TIMEOUT", "10")


def download_and_save(dataset_name, save_dir):
    """
    下载 Hugging Face 数据集并保存到指定目录。

    :param dataset_name: 数据集名称，例如 "drt/complex_web_questions"
    :param save_dir: 保存数据集的根目录
    """
    os.makedirs(save_dir, exist_ok=True)

    print(f"开始下载数据集: {dataset_name}")
    
    # 直接使用 load_dataset 下载数据集
    try:
        dataset = load_dataset(dataset_name, trust_remote_code=True)
    except Exception as e:
        print(f"下载失败: {e}")
        print(f"尝试不使用远程代码下载...")
        dataset = load_dataset(dataset_name)

    # 遍历数据集的每个拆分并保存为文件（带进度条）
    for split, data in tqdm(dataset.items(), desc=f"处理 {dataset_name}"):
        save_path = os.path.join(save_dir, f"{dataset_name.replace('/', '_')}_{split}.json")
        print(f"保存 {split} 拆分到 {save_path}")
        data.to_json(save_path)

    print(f"数据集 {dataset_name} 下载并保存完成！")


def convert_parquet_to_json(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # 定义训练、验证和测试的文件组
    file_groups = {
        "train": ["train-00000-of-00002.parquet", "train-00001-of-00002.parquet"],
        "validation": ["validation-00000-of-00001.parquet"],
        "test": ["test-00000-of-00002.parquet", "test-00001-of-00002.parquet"]
    }

    for split, files in file_groups.items():
        combined_data = []
        for file in files:
            file_path = os.path.join(input_dir, file)
            if os.path.exists(file_path):
                df = pd.read_parquet(file_path)
                combined_data.extend(df.to_dict(orient="records"))

        # 将合并的数据写入 JSON 文件
        output_file = os.path.join(output_dir, f"{split}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(combined_data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # 数据集名称列表
    datasets_to_download = [
        "ml1996/webqsp",
        # 如果有其他数据集，可以在这里添加
    ]

    # 保存数据集的根目录（以项目根目录的 datasets 目录为准）
    root_save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets"))

    for dataset_name in datasets_to_download:
        download_and_save(dataset_name, root_save_dir)

    # 转换 parquet 文件为 JSON 格式
    input_directory = "datasets/webqsp"
    output_directory = "datasets/webqsp/json_output"
    convert_parquet_to_json(input_directory, output_directory)
