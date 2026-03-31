#!/usr/bin/env python3
"""
数据准备脚本：将 CWQ 数据转换为 slime 训练所需格式

输入格式 (cwq_train.json):
{
  "id": "cwq_0",
  "question": "What state is home to...",
  "answers": ["Washington, D.C.", ...],
  "topic_entity": {"m.03d0l76": "George Washington..."}
}

输出格式 (训练数据JSON):
{
  "question": "What state is home to...",
  "metadata": "{\"topic_entity\": {...}, \"answers\": [...]}"
}

注意：question字段存储原始问题，完整prompt在generate函数内部构建
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, List


def process_single_example(example: Dict[str, Any]) -> Dict[str, Any]:
    """处理单个样本，转换为slime训练格式
    
    Args:
        example: 原始CWQ格式的样本
        
    Returns:
        slime训练格式的样本
    """
    # 提取必要字段
    question = example.get("question", "")
    answers = example.get("answers", [])
    topic_entity = example.get("topic_entity", {})
    example_id = example.get("id", "")
    
    # 确保answers是列表
    if isinstance(answers, str):
        answers = [answers]
    
    # 构建metadata（JSON字符串）
    metadata = {
        "id": example_id,
        "topic_entity": topic_entity,
        "answers": answers,
    }
    
    return {
        "question": question,
        "metadata": json.dumps(metadata, ensure_ascii=False),
    }


def convert_dataset(
    input_path: str,
    output_path: str,
    limit: int = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """转换整个数据集
    
    Args:
        input_path: 输入JSON文件路径
        output_path: 输出JSON文件路径
        limit: 可选，限制处理的样本数量
        verbose: 是否打印详细信息
        
    Returns:
        转换统计信息
    """
    # 读取输入数据
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if verbose:
        print(f"Loaded {len(data)} examples from {input_path}")
    
    # 限制数量
    if limit is not None and limit > 0:
        data = data[:limit]
        if verbose:
            print(f"Limited to {len(data)} examples")
    
    # 转换每个样本
    converted = []
    skipped = 0
    
    for idx, example in enumerate(data):
        try:
            converted_example = process_single_example(example)
            
            # 验证转换结果
            if not converted_example["question"]:
                skipped += 1
                if verbose:
                    print(f"  Skipped example {idx}: empty question")
                continue
                
            converted.append(converted_example)
            
        except Exception as e:
            skipped += 1
            if verbose:
                print(f"  Skipped example {idx}: {e}")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    # 写入输出（JSONL格式：每行一个JSON对象）
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in converted:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    stats = {
        "input_path": input_path,
        "output_path": output_path,
        "total_input": len(data),
        "total_output": len(converted),
        "skipped": skipped,
    }
    
    if verbose:
        print(f"\nConversion complete:")
        print(f"  Input: {stats['total_input']} examples")
        print(f"  Output: {stats['total_output']} examples")
        print(f"  Skipped: {stats['skipped']} examples")
        print(f"  Saved to: {output_path}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Convert CWQ dataset to slime training format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="../datasets/cwq_train.json",
        help="Path to input CWQ JSON file"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="../datasets/cwq_train_prepared.jsonl",
        help="Path to output JSONL file (one JSON object per line)"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Limit number of examples to process (for testing)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print verbose output"
    )
    
    args = parser.parse_args()
    
    # 处理相对路径
    script_dir = Path(__file__).parent
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.is_absolute():
        input_path = script_dir / input_path
    if not output_path.is_absolute():
        output_path = script_dir / output_path
    
    # 执行转换
    stats = convert_dataset(
        str(input_path),
        str(output_path),
        limit=args.limit,
        verbose=args.verbose or True  # 默认verbose
    )
    
    print(f"\n✓ Successfully converted {stats['total_output']} examples")


if __name__ == "__main__":
    main()
