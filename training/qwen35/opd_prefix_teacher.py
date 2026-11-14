"""OPD 启动器：教师用自己的提示前缀打分（学生不变），再调用 MedicalGPT 的 opd_training.main()。

为什么（2026-11-14 教师检查发现）：
  学生（SFT / DPO / 部署）用的 qwen3_5_nothink 模板在 assistant 开头没有空思考块；Qwen3.5-27B 看到这种开头会先
  输出 "<think> Thinking Process: ..."。TRL GKDTrainer 让教师和学生吃同一串 input_ids，教师于是在“思考模式”下给
  学生的直接回答打分，蒸馏信号整体错位。这里只改教师那一侧：
    教师提示 = 学生提示 [+ 系统提示末尾追加 TEACHER_SYSTEM_EXTRA] + "<think>\\n\\n</think>\\n\\n"
  回答 token 两边完全相同，按回答位置对齐后算同样的 generalized JSD。
  TEACHER_SYSTEM_EXTRA（可选）= 只给教师看的额外规则（特权信息蒸馏）：学生看不到这条规则，但会学到教师有这条规则时的说法。

用法（在 MedicalGPT 目录下）：python <本文件> <opd_training.py 的全部参数>
"""
import os
import sys

import torch

sys.path.insert(0, os.getcwd())  # MedicalGPT 根目录，opd_training 用 training.template 这种包路径
import training.opd_training as opd  # noqa: E402

THINK_OFF = "<think>\n\n</think>\n\n"
EXTRA = os.environ.get("TEACHER_SYSTEM_EXTRA", "").strip()


class PrefixTeacherGKDTrainer(opd.GKDTrainer):
    _cache: dict = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # GKDTrainer 在开梯度检查点时把生成的 use_cache 设为 False，学生生成每个 token 都重算整段（2026-11-14 实测约 150ms/token）。
        # 生成时 unwrap_model_for_generation 本来就会临时关掉梯度检查点，打开 KV 缓存是安全的。
        self.generation_config.use_cache = True
        self.generation_kwargs["use_cache"] = True

    def _teacher_prompt_ids(self, prompt_ids: list[int]) -> list[int]:
        key = tuple(prompt_ids)
        if key not in self._cache:
            tok = self.processing_class
            text = tok.decode(prompt_ids, skip_special_tokens=False)
            if EXTRA:
                cut = text.find("<|im_end|>")  # 第一个 <|im_end|> = 系统提示结尾
                if cut < 0:
                    raise ValueError("提示里找不到系统段，无法追加 TEACHER_SYSTEM_EXTRA")
                text = text[:cut].rstrip("\n") + "\n" + EXTRA + text[cut:]
            if not text.endswith(THINK_OFF):
                text += THINK_OFF
            self._cache[key] = tok(text, add_special_tokens=False)["input_ids"]
        return self._cache[key]

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """2026-11-14 第一次正式训练在第 ~150 步 OOM：数据里的参考回答（off-policy 批次）没有截断，4 条一起算 JSD 要 10.6GB。
        现在：每条只取回答的前 max_new_tokens 个 token（与学生生成上限一致），重新右填充成一批做学生前向；
        JSD 逐条计算后求和，再除以本批有效 token 数（GA=1 时等价于原版按全局 token 数归一）。"""
        ids, mask, labels = inputs["input_ids"], inputs["attention_mask"], inputs["labels"]
        max_comp = int(self.args.max_new_tokens)
        rows = []
        for i in range(ids.size(0)):
            pos = mask[i].nonzero().squeeze(-1)  # 有效 token 的位置（左右 padding 都适用）
            v_ids, v_lab = ids[i, pos], labels[i, pos]
            comp = (v_lab != -100).nonzero().squeeze(-1)
            if comp.numel() == 0:
                continue
            b = comp[0].item()  # 回答起点
            end = min(v_ids.size(0), b + max_comp)
            rows.append((v_ids[:end], v_lab[:end], b))
        if not rows:
            student_outputs = model(input_ids=ids, attention_mask=mask)
            loss = student_outputs.logits.sum() * 0.0
            return (loss, student_outputs) if return_outputs else loss
        L = max(r[0].size(0) for r in rows)
        pad_id = self.processing_class.pad_token_id or 0
        new_ids = torch.full((len(rows), L), pad_id, dtype=ids.dtype, device=ids.device)
        new_mask = torch.zeros((len(rows), L), dtype=mask.dtype, device=ids.device)
        new_lab = torch.full((len(rows), L), -100, dtype=labels.dtype, device=ids.device)
        for r, (v_ids, v_lab, _) in enumerate(rows):  # 右填充：有效 token 的位置编号从 0 开始，和单条推理一致
            n = v_ids.size(0)
            new_ids[r, :n], new_mask[r, :n], new_lab[r, :n] = v_ids, 1, v_lab
        student_outputs = model(input_ids=new_ids, attention_mask=new_mask)

        self.teacher_model.eval()
        total, count = None, 0
        for r, (v_ids, v_lab, b) in enumerate(rows):
            n = v_ids.size(0)
            t_prompt = self._teacher_prompt_ids(v_ids[:b].tolist())
            t_ids = torch.cat([torch.tensor(t_prompt, dtype=v_ids.dtype, device=v_ids.device), v_ids[b:]])[None]
            with torch.no_grad():
                t_logits = self.teacher_model(input_ids=t_ids).logits[:, len(t_prompt) - 1:-1]
            lab = v_lab[b:n][None]
            part = self.generalized_jsd_loss(
                student_logits=student_outputs.logits[r:r + 1, b - 1:n - 1], teacher_logits=t_logits,
                labels=lab, beta=self.beta, reduction="sum")
            total = part if total is None else total + part
            count += int((lab != -100).sum())
            del t_logits
        loss = total / max(count, 1)
        return (loss, student_outputs) if return_outputs else loss


class _ResolvingParser(opd.HfArgumentParser):
    """MedicalGPT 的 ModelArguments 和 TRL 1.17 的 SFTConfig（GKDConfig 的父类）都定义了 trust_remote_code，
    argparse 默认遇到重名直接报错。改成 resolve：命令行上只剩一个同名选项，解析结果按字段名分给两个 dataclass，取值相同。"""

    def __init__(self, dataclass_types, **kwargs):
        import dataclasses
        seen, dup = {}, []
        self._dup_counts = {}
        for t in dataclass_types:
            for f in dataclasses.fields(t):
                if f.name in seen:
                    dup.append(f"{f.name}（{seen[f.name]} / {t.__name__}）")
                    self._dup_counts[f.name] = self._dup_counts.get(f.name, 1) + 1
                seen.setdefault(f.name, t.__name__)
        if dup:
            print("[opd_prefix_teacher] 重名参数合并为同一个值：" + "，".join(dup))
        kwargs.setdefault("conflict_handler", "resolve")
        super().__init__(dataclass_types, **kwargs)

    def parse_known_args(self, args=None, namespace=None):
        # parse_args_into_dataclasses 每分完一个 dataclass 就 delattr 它的字段；重名字段要留到最后一个 dataclass 再删
        if namespace is None:
            import argparse
            left = dict(self._dup_counts)

            class _KeepDupNamespace(argparse.Namespace):
                def __delattr__(ns, key):
                    if left.get(key, 1) > 1:
                        left[key] -= 1
                        return
                    super().__delattr__(key)

            namespace = _KeepDupNamespace()
        return super().parse_known_args(args, namespace)


opd.GKDTrainer = PrefixTeacherGKDTrainer
opd.HfArgumentParser = _ResolvingParser
if __name__ == "__main__":
    print(f"[opd_prefix_teacher] 教师前缀：空思考块；额外规则：{EXTRA or '无'}")
    opd.main()
