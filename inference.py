from dataclasses import dataclass
from typing import Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABEL_MAPPING = {
    0: "negative",
    1: "positive",
}

@dataclass
class InferenceConfig:
    model_dir: str = "models/imdb-distilroberta"
    max_length: int = 256

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        print("[inference] MPS is available, using Apple GPU (MPS).")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("[inference] CUDA is available, using NVIDIA GPU.")
        return torch.device("cuda")
    else:
        print("[inference] No GPU found, using CPU.")
        return torch.device("cpu")
    
def load_model_and_tokenizer(config: InferenceConfig):
    print(f"Loading tokenizer and model from: {config.model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(config.model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(config.model_dir)

    device = get_device()
    model.to(device)
    model.eval() # 推理模式，关闭 dropout 等

    return tokenizer, model, device

def predict(
    text: str,
    tokenizer,
    model,
    device: torch.device,
    config: InferenceConfig,
) -> Dict[str, Any]:
    """
    返回字典：
    {
        "label": "positive" / "negative",
        "score": 0.97,                    # 预测类别的置信度
        "probs": {"negative": 0.03, "positive": 0.97}
    }
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    
    text = text.strip()
    if not text:
        raise ValueError("text is empty")
    
    inputs = tokenizer(
        text,
        truncation=True,
        max_length=config.max_length,
        padding=False, # 预测时单条文本不必padding
        return_tensors="pt", # 返回PyTorch tensor
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        # 函数调用时：
        # *args：把一个序列按“位置参数“展开
        # **kwargs：把一个字典按“关键字参数“展开，以 key 作为参数名，value 作为对应的参数值
        outputs = model(**inputs)
        # logits = 模型对每个类别给出的“原始分数“（还没变成概率之前的值）
        logits = outputs.logits

    probs = torch.softmax(logits, dim=-1)[0]
    score, pred_id = torch.max(probs, dim=-1)

    pred_id_int = int(pred_id.item())
    label = LABEL_MAPPING.get(pred_id_int, str(pred_id_int))

    probs_dict = {
        LABEL_MAPPING.get(i, str(i)): float(probs[i].item())
        for i in range(probs.shape[0])
    }

    return {
        "label": label,
        "score": float(score.item()),
        "probs": probs_dict,
    }

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Sentiment inference with fine-tuned DistilRoBERTa on IMDb."
    )

    parser.add_argument(
        "--text",
        type=str,
        help="Input text to classify (if omitted, will enter interactive mode)."
    )

    args = parser.parse_args()

    config = InferenceConfig()
    tokenizer, model, device = load_model_and_tokenizer(config)

    if args.text:
        result = predict(args.text, tokenizer, model, device, config)
        print("\n=== Prediction ===")
        print(f"Text   : {args.text}")
        print(f"Label  : {result['label']}")
        print(f"Score  : {result['score']:.4f}")
        print(f"Probs  : {result['probs']}")
        return
    
    print("\n Enter text to classify sentiment (empty line to quit):")
    while True:
        try:
            text = input(">> ").strip()
        except EOFError:
            break

        if not text:
            print("Bye.")
            break

        try:
            result = predict(text, tokenizer, model, device, config)
        except Exception as e:
            print(f"Error: {e}")
            continue

        print(f"  -> Label : {result['label']}  (score={result['score']:.4f})")
        print(f"     Probs : {result['probs']}")

if __name__ == "__main__":
    main()