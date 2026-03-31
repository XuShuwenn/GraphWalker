#!/usr/bin/env python3
"""
校验 convert_torch_dist_to_hf.py 转换后的 HuggingFace 格式 checkpoint 是否能正常加载。

用法:
  python ./graphwalker/scripts/verify_hf_checkpoint.py --path /path/to/converted/iter_0000059
  python ./graphwalker/scripts/verify_hf_checkpoint.py --path /path/to/iter_0000059 --do-forward  
"""
import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Verify converted HF checkpoint loads correctly.")
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the converted HF checkpoint directory (e.g. saves_hf/.../iter_0000059)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to load model on (default: cuda:0). Use cpu for no-GPU check.",
    )
    parser.add_argument(
        "--do-forward",
        action="store_true",
        help="Run a short forward pass to verify inference works.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=8,
        help="When --do-forward, max new tokens to generate (default: 8).",
    )
    args = parser.parse_args()

    path = args.path.rstrip("/")
    if not path or not os.path.isdir(path):
        print(f"[FAIL] Not a directory: {path}")
        sys.exit(1)

    # Lazy import so --help works without transformers
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    print(f"Verifying HF checkpoint: {path}")
    print("-" * 60)

    # 1. Config
    try:
        config = AutoConfig.from_pretrained(path, trust_remote_code=True)
        print(f"[OK] Config loaded (model_type={getattr(config, 'model_type', '?')})")
    except Exception as e:
        print(f"[FAIL] Config load: {e}")
        sys.exit(1)

    # 2. Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        print(f"[OK] Tokenizer loaded (vocab_size={getattr(tokenizer, 'vocab_size', '?')})")
    except Exception as e:
        print(f"[FAIL] Tokenizer load: {e}")
        sys.exit(1)

    # 3. Model
    try:
        model = AutoModelForCausalLM.from_pretrained(
            path,
            trust_remote_code=True,
            torch_dtype="auto",
        )
        model = model.to(args.device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[OK] Model loaded on {args.device} ({n_params / 1e9:.2f}B params)")
    except Exception as e:
        print(f"[FAIL] Model load: {e}")
        sys.exit(1)

    # 4. Optional forward / generation
    if args.do_forward:
        try:
            import torch
            model.eval()
            dummy = tokenizer("Hello", return_tensors="pt").to(args.device)
            with torch.no_grad():
                out = model.generate(
                    **dummy,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id,
                )
            gen_text = tokenizer.decode(out[0][dummy["input_ids"].shape[1] :], skip_special_tokens=True)
            print(f"[OK] Forward + generate ({args.max_new_tokens} tokens): {repr(gen_text[:50])}...")
        except Exception as e:
            print(f"[FAIL] Forward/generate: {e}")
            sys.exit(1)

    print("-" * 60)
    print("Verification passed. Checkpoint is loadable.")
    sys.exit(0)


if __name__ == "__main__":
    main()
