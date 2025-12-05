from dataclasses import dataclass # 从标准库 dataclasses 模块导入 dataclass 装饰器
from typing import Tuple

import torch
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, PreTrainedTokenizerBase

@dataclass # 帮我自动生成构造函数、打印方法等的‘数据类‘装饰器，用于存配置、参数、简单数据结构等
class TrainConfig(): # 更优雅的配置类写法
    # 默认模型：distilroberta-base
    model_name: str = "distilroberta-base"

    # 默认数据集：imdb
    dataset_name: str = "imdb"

    max_length: int = 256

    seed: int = 666

def get_device() -> torch.device:
    """获取训练设备，优先使用 Apple GPU(mps), 其次 CUDA，最后 CPU"""
    if torch.backends.mps.is_available():
        print("MPS is available, using Apple GPU (MPS).")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("CUDA is available, using Nvidia GPU.")
        return torch.device("cuda")
    else:
        print("no GPU found, using CPU")
        return torch.device("cpu")
    
# Tokenizer所做的工作：按照相对应的模型
# 1、规范文本（如转小写，去掉奇怪字符等）
# 2、将一句话切成token或子词
# 3、将每个token映射成整数ID
# 4、处理长度 & 掩码
# 5、整理成模型要的字段格式
# tokenizer = 固定的算法框架 + “在大语料上训练出来的词表/规则“

# tokenizer把文本变成一串整数 ID
# Embedding层就是将这些整数ID变成“有意义的向量“，供Transformer使用
def load_data_and_tokenizer(
    config: TrainConfig
) -> Tuple[DatasetDict, DatasetDict, PreTrainedTokenizerBase]:
    torch.manual_seed(config.seed)

    print(f"Loading dataset: {config.dataset_name}")
    raw_datasets = load_dataset(config.dataset_name)
    print(raw_datasets)

    example = raw_datasets["train"][0]
    print("\nRaw example from train dataset:")
    print(example)

    print(f"\n Loading tokenizer: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length = config.max_length,
        )
    
    print("\n Tokenizing dataset...")
    # 将 raw_datasets 中的样本，按批次读出来，每一批交给 tokenize_function
    # tokenize_function 负责用对应模型的 tokenizer，把这一批里的 text 列转为 input_ids 和 attention_mask
    # 并与原批次合并，再丢掉text列
    # map 是一种函数式风格的操作，不直接修改原来的 Data，而是基于原数据构建新的数据集
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"], # tokenized处理过后，text已没有作用
        desc="Tokenizing", # 给进度条看的说明文字
    )

    if "label" in tokenized_datasets["train"].column_names:
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    tokenized_datasets.set_format( # 就地修改视图设置
        type="torch", # 自动将它们以torch.Tensor形式返回（底层数据没变，返回形式变了）
        columns=["input_ids", "attention_mask", "labels"], # 以后从 tokenized_datasets 取数据时，只给这三列
    )

    print("\n Tokenization done. Tokenized dataset:")
    print(tokenized_datasets)

    print("\n One tokenized example from train dataset:")
    print(tokenized_datasets["train"][0])

    return raw_datasets, tokenized_datasets, tokenizer

def main():
    config = TrainConfig()

    print("==== TrainConfig ====")
    print(config)

    device = get_device()
    print("Using device: ", device)

    raw_datasets, tokenized_datasets, tokenizer = load_data_and_tokenizer(config)

    print("\n🎉 Pipeline check finished.")
    print("You now have:")
    print("  - raw_datasets      (with text + label)")
    print("  - tokenized_datasets(input_ids + attention_mask + labels)")
    print("  - tokenizer         (distilroberta-base)")
    print("\nNext step: we will add model + TrainingArguments + Trainer on top of this.\n")

# 直接执行时，内置变量__name__为__main__
# 通过导入执行时，内置变量__name__为模块名 train（无下划线）
if __name__ == "__main__":
    main()   