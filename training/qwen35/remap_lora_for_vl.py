"""把按 Qwen3_5ForCausalLM 训出的 LoRA adapter 改成 Qwen3_5ForConditionalGeneration 的参数名。

为什么需要：MedicalGPT 的 SFT / DPO 用 AutoModelForCausalLM 加载 Qwen3.5，只建文本部分，参数名是
model.layers.*；merge_peft_adapter.py 看到 config 里是 ConditionalGeneration，会按多模态结构加载
（model.language_model.layers.*）。名字对不上时 PEFT 只给警告、LoRA 权重保持初始值（B=0），
合并出来就是原版模型。这里把 key 改名，并把 target_modules 限定在文本层，避免给视觉塔挂空 LoRA。

用法：python remap_lora_for_vl.py <adapter_dir> <out_dir>
合并后用 --check <base_dir> <merged_dir> <adapter_dir> 核对某几层的增量是否等于 B@A*alpha/r。
"""
import json
import os
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

PRE = "base_model.model.model."


def remap(src: str, dst: str) -> None:
    os.makedirs(dst, exist_ok=True)
    sd = load_file(os.path.join(src, "adapter_model.safetensors"))
    out = {}
    for k, v in sd.items():
        if k.startswith(PRE + "layers."):
            k = PRE + "language_model." + k[len(PRE):]
        elif not k.startswith(PRE + "language_model.layers."):
            raise SystemExit(f"意外的 key：{k}")
        out[k] = v
    save_file(out, os.path.join(dst, "adapter_model.safetensors"), metadata={"format": "pt"})
    cfg = json.load(open(os.path.join(src, "adapter_config.json")))
    names = cfg["target_modules"]
    if isinstance(names, list):
        cfg["target_modules"] = r".*language_model\.layers\.\d+\..*\.(?:%s)" % "|".join(sorted(names))
    json.dump(cfg, open(os.path.join(dst, "adapter_config.json"), "w"), indent=2, ensure_ascii=False)
    for f in ("README.md",):
        if os.path.isfile(os.path.join(src, f)):
            shutil.copy2(os.path.join(src, f), dst)
    print(f"改名 {len(out)} 个 key → {dst}；target_modules = {cfg['target_modules']}")


def _tensor(model_dir: str, key: str, raw: bool = False) -> torch.Tensor:
    idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    with safe_open(os.path.join(model_dir, idx[key]), "pt") as f:
        t = f.get_tensor(key)
        return t if raw else t.float()


def check(base: str, merged: str, adapter: str) -> None:
    cfg = json.load(open(os.path.join(adapter, "adapter_config.json")))
    scale = cfg["lora_alpha"] / cfg["r"]
    sd = load_file(os.path.join(adapter, "adapter_model.safetensors"))
    mods = sorted({k.split(".lora_")[0] for k in sd})
    picks = [m for m in mods if ".layers.3." in m or ".layers.0." in m][:4] + mods[-1:]
    worst = 1.0
    for m in picks:
        wkey = m.replace(PRE, "model.").replace("model.layers.", "model.language_model.layers.") + ".weight"
        a, b = sd[m + ".lora_A.weight"].float(), sd[m + ".lora_B.weight"].float()
        want = b @ a * scale
        w0, w1 = _tensor(base, wkey, raw=True), _tensor(merged, wkey, raw=True)
        got = w1.float() - w0.float()
        cos = torch.nn.functional.cosine_similarity(got.flatten(), want.flatten(), dim=0).item()
        # 理想合并 = W+ΔW 在 fp32 里算好再转回存盘精度；逐元素一致率比余弦可靠
        ideal = (w0.float() + want).to(w1.dtype)
        match = (ideal == w1).float().mean().item()
        changed = (w1 != w0).float().mean().item()
        worst = min(worst, match)
        print(f"{wkey}: |ΔW|={got.norm():.4f} 期望 {want.norm():.4f} 余弦 {cos:.3f} 与理想合并一致 {match:.4f} 有变化的元素 {changed:.3f}")
    # 2026-11-14：DPO（lr 5e-6）的 ΔW 中位只有 W 的千分之一，bf16 舍入后约 84% 元素不变，余弦只剩 0.4~0.6，
    # 但逐元素与理想合并 100% 一致——合并没错，是 bf16 存盘的精度上限（运行时挂 adapter 的 margin 约保留 73%）。
    # 合并失败（LoRA 没加上）时，凡是 ΔW 超过半个舍入步长的元素都会对不上，一致率明显低于 0.99。
    ok = worst > 0.99
    print("合并核对：" + ("通过" if ok else "失败"))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    if sys.argv[1] == "--check":
        check(*sys.argv[2:5])
    else:
        remap(sys.argv[1], sys.argv[2])
