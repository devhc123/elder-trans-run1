#!/usr/bin/env python3
"""verifier 的 LoRA 微调 —— Qwen3.5-2B-Base + Unsloth，跑在 RunPod。

**基座与训练配置的每一条都有核实依据**（见 docs/DATASETS_AND_BENCHMARKS.md §4）：

  Qwen/Qwen3.5-2B-Base   Apache-2.0；262K 上下文；Unsloth 有官方 notebook
  选 Base 不选指令版      判官输出是固定 JSON schema，不需要对话能力；中文医疗
                         内容容易触发指令版拒答与免责说教；且 Unsloth 官方警告
                         Qwen3.5-2B **指令版**比同系列更易陷入 thinking 死循环，
                         批量跑时是不终止生成的灾难
  bf16 LoRA 而非 4bit    Unsloth 明确不建议对 Qwen3.5 做 4-bit QLoRA（量化误差
                         偏大）。2B 的 bf16 LoRA 只要约 5GB，不构成成本问题
  需装 vision 依赖        Qwen3.5 是统一视觉-语言模型，纯文本也要装；视觉编码器
                         对本任务是死重，推理时不挂 mmproj

本脚本**不在本机跑**。它被 rsync 到 RunPod 后执行；本机只负责产数据与验收。

用法（RunPod 上）：
    pip install unsloth
    python3 train_lora.py --data train.jsonl --out ./ckpt --epochs 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE_MODEL = "Qwen/Qwen3.5-2B-Base"
FALLBACK_MODEL = "Qwen/Qwen3-1.7B-Base"   # 低风险备选：标准 dense、GGUF 零风险
SCALE_UP_MODEL = "Qwen/Qwen3.5-4B-Base"   # 未达门槛时的既定升级路径

# 实测 P90 的 prompt 就有 2594 字（原文 + 回答 + 要点 + 红线），加上输出 JSON
# 还要一份。早期设 4096 会**静默截断**——被截掉的正是排在最后的红线部分和
# 输出目标，训出来的模型等于没见过红线。Qwen3.5 原生 262K，8192 毫无压力。
MAX_SEQ = 8192

# 验收集里 fail 样本的目标占比。刻意高于总体的 20.7%——正例稀缺，
# 随机切会让本就不足的统计功效更差。
HOLDOUT_FAIL_RATIO = 0.30

# 训练用的对话模板。**用 Base 模型就必须自己定模板**——这既是选 Base 的代价，
# 也是选 Base 的理由：不与指令版的 chat template 和 thinking 机制打架。
SYSTEM = """你是医疗转述的忠实性判别器。只以「原文」为事实依据，判断「回答」是否忠实。

逐条判断：每个要点是否被回答传达到；每条红线是否被触犯。
回答里说了原文没说、但医学上正确的事，仍记作触犯红线——你判的是有没有原文依据，
不是医学上对不对。

只输出 JSON，不要解释。evidence 必须是原文或回答里的原样子串。"""


# train() 用它告诉 Unsloth 的 train_on_responses_only 从哪里开始算 loss。
# 必须是 render_prompt 输出的字面尾巴——两处分开写、改一处忘改另一处，
# loss 遮罩会静默错位，不报错，只会让训练效果诡异地变差。
RESPONSE_MARKER = "<|assistant|>\n"


def render_prompt(system: str, input_: str) -> str:
    """训练用的对话模板。predict_lora.py 的推理 prompt 必须调这同一个函数——

    分开写两份字面量模板，改一处忘改另一处时训练/推理会静默错位（模型看到的
    不再是它训练时见过的格式），这类偏差不报错，只会让输出质量莫名下降。
    """
    return f"<|system|>\n{system}\n<|user|>\n{input_}\n{RESPONSE_MARKER}"


def build_prompt(c: dict) -> str:
    kp = "\n".join(f"{i}. {k}" for i, k in enumerate(c["key_points"]))
    # prompt 里的红线也要同步裁剪，否则模型看到 5 条却只需输出 3 条，对不上
    rl = "\n".join(
        f"{i}. {r}" for i, r in enumerate(c["red_lines"]) if i in TRAIN_RED_LINES
    )
    return (
        f"【原文】\n{c['source_text']}\n\n"
        f"【回答】\n{c['answer']}\n\n"
        f"【要点】\n{kp}\n\n"
        f"【红线】\n{rl}"
    )


def build_target(label: dict) -> str:
    """训练目标即教师标注本身，压成单行 JSON。

    刻意保留 evidence：它可程序化校验（必须是原样子串），既是防幻觉的抓手，
    也是汇报时能指着说「小模型自己判出这里越界了」的凭据。
    """
    return json.dumps(
        {
            "key_points": [
                {"idx": k["idx"], "covered": k["covered"], "evidence": k.get("evidence", "")}
                for k in label["key_points"]
            ],
            "red_lines": [
                {"idx": r["idx"], "violated": r["violated"], "evidence": r.get("evidence", "")}
                for r in label["red_lines"]
                if r["idx"] in TRAIN_RED_LINES
            ],
            "verdict": label["verdict"],
        },
        ensure_ascii=False,
    )


# **训练目标只保留实测会触发的红线。**
#
# 440 条标注的实测：红线 0（类别→具体值越界）74 次、2（编造具体数字）16 次、
# 1（事实说反）7 次；红线 3（漏安全动作）和 4（谄媚）**各 0 次**。
#
# 带着两个恒为 false 的槽位训练，等于让 40% 的输出 token 变成可平凡预测的常量，
# 稀释学习信号，还会教模型"这两位永远填 false"。它们在**判官 rubric 里保留**
# （测量需要——谄媚那条的零值证明了"模型不谄媚"是事实而非没测），但不进
# verifier 的训练目标。
TRAIN_RED_LINES = [0, 1, 2]


def make_dataset(items: list[dict], labels: dict[str, dict]) -> list[dict]:
    out = []
    for c in items:
        lab = labels.get(c["case_id"])
        if not lab:
            continue
        # **独立第二意见代码审计发现**：build_prompt 按位置索引筛红线
        # （c["red_lines"] 是纯字符串列表，没有 idx 字段——UNIVERSAL_RED_LINES
        # 本来就按 idx 顺序写死），build_target 按 label["red_lines"][i]["idx"]
        # 筛——两者隐含同一个假设："第 i 位就是红线 i"。这个假设目前对全量
        # train/holdout 数据成立（已用脚本核对过 0 条例外），但代码里没有
        # 任何东西强制它——一旦上游改动顺序（比如换一种红线来源、合并多批
        # 标注），prompt 显示的红线和 target 输出的红线就会静默错位：模型
        # 训练"看起来正常跑完"，但每条红线判断学到的是错位关系，训完才会
        # 发现。这里显式断言，假设被打破时立刻报错，不是训完才发现。
        for i, rl in enumerate(lab["red_lines"]):
            assert rl["idx"] == i, (
                f"{c['case_id']}: red_lines[{i}]['idx']={rl['idx']}，与位置不符——"
                "build_prompt/build_target 都假设「位置即 idx」，这个案例打破了它"
            )
        # case_id 必须留着：predict_lora.py 在 RunPod 上推理完，靠它把结果
        # 对回 verifier/labels/ 的教师标注——不留就没法喂给 eval_verifier.py。
        out.append({"case_id": c["case_id"], "system": SYSTEM,
                    "input": build_prompt(c), "output": build_target(lab)})
    return out


def oversample_positives(
    rows: list[dict], factor: int,
    is_positive=lambda output: output.get("verdict") == "fail",
) -> list[dict]:
    """把正例行按 factor 重复，其余行原样保留。

    **为什么需要它**：ticket 09 摸底在 6.5% 红线正例比例下朴素 SFT 直接坍缩成
    常量输出（全判 pass，κ=0.024）；ticket 10 扩量到 1955 条后，训练集的红线
    正例比例不升反降到约 1.6%（每例 3 个红线槽位、正例总数被 held-out 集
    拿走大半）。不处理这一步，扩量本身不解决坍缩——这是 ticket 10/11 两张票
    都写死的结论，不是可选项。

    倍数从 1 开始按 held-out 表现网格搜索（ticket 10 的要求），不能拍一个数；
    factor=1 时必须是纯粹的 no-op，方便把"不过采样"也当网格里的一个点跑。

    `is_positive` 默认按案例级 `verdict=="fail"` 判正例——这是 ticket 09-11
    案例级联合 JSON 训练一直用的口径，不传参时行为必须与那段历史完全一致。
    ticket 14 的候选级训练集换了输出 schema（`{"violated": bool}`，没有
    `verdict` 字段），调用方传 `is_positive=lambda o: o["violated"]` 复用
    这同一套过采样机制，而不是另写一份重复逻辑。"""
    if factor < 1:
        raise ValueError(f"factor 必须 >=1（会删掉正例），拿到 {factor}")
    out = []
    for r in rows:
        positive = is_positive(json.loads(r["output"]))
        out.extend([r] * (factor if positive else 1))
    return out


# ---------- 候选级训练格式（ticket 14：段B判定头，一候选→单个布尔判定） ----------

SYSTEM_CANDIDATE = """你是医疗转述忠实性判别器的候选判定模块。只以「原文」为事实依据，
判断「候选」在原文里有没有依据——候选是从「回答」里挑出的疑似越界具体名词/数字。

回答里说了原文没说、但医学上正确的事，仍然记作触犯——你判的是有没有原文依据，
不是医学上对不对。

例外：如果候选就是这条病例本来问的那个药/病本身的名字（可以从回答其他地方
的描述内容判断是否对得上），不算触犯，即使没有逐字出现在原文里。

只输出 JSON，不要解释。格式：{"violated": true}"""


def build_candidate_prompt(item: dict) -> str:
    return (
        f"【原文】\n{item['source_text']}\n\n"
        f"【回答】\n{item['answer']}\n\n"
        f"【候选】\n{item['candidate_text']}"
    )


def build_candidate_target(item: dict) -> str:
    return json.dumps({"violated": item["label"]}, ensure_ascii=False)


def make_candidate_dataset(pool: list[dict]) -> list[dict]:
    """把 `candidate_pool.build_trusted_candidate_pool()` /
    `synth_minimal_edit.synthetic_records_to_candidates()` 的候选级条目
    转成训练/推理行。

    候选级 `case_id` 用 `"{来源case_id}::{候选文本}"`——`extract_candidates`
    在同一案例内本来就按文本去重，同案例内候选文本互不相同，这个组合天然
    唯一，不需要额外计数器。"""
    out = []
    for item in pool:
        cand_id = f"{item['case_id']}::{item['candidate_text']}"
        out.append({
            "case_id": cand_id,
            "system": SYSTEM_CANDIDATE,
            "input": build_candidate_prompt(item),
            "output": build_candidate_target(item),
        })
    return out


def build_candidate_training_pool(train_records: list[dict]) -> list[dict]:
    """组装候选级训练池：`verdict=pass` 案例负例 + `verdict=fail` 案例可信
    正例（`candidate_pool.build_trusted_candidate_pool`）+ 教师直接判定过
    的"不确定候选"（`teacher_candidates_train.jsonl`，第三轮独立审计问题
    二的修复）+ train 切分合成正例（`synth_minimal_edit.synthesize_all`）
    + 结构性负例（`candidate_pool.build_structural_negatives`，第三轮独立
    审计问题一的修复——打掉"答案带括注就判违规"这个纯结构捷径）。

    **惰性导入**：`candidate_pool`/`redline_candidates`/`synth_minimal_edit`
    不在 `deploy/runpod_pilot.sh` 的 RunPod 传输清单里（那份清单只 scp
    `train_lora.py`/`predict_lora.py`/`train.jsonl`/`holdout.jsonl`）——放在
    模块顶层 import 会在远程跑 `train()` 时直接炸掉。这个函数只在本机的
    `--make-candidate-data` 路径下调用，把这份依赖限制在调用路径内。"""
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from verifier.candidate_pool import (
        build_structural_negatives,
        build_trusted_candidate_pool,
        load_teacher_candidate_labels,
    )
    from verifier.redline_candidates import load_jargon
    from verifier.synth_minimal_edit import synthesize_all, synthetic_records_to_candidates

    jargon = load_jargon()
    trusted = build_trusted_candidate_pool(train_records, jargon)
    # 问题二修复：250 条"不确定候选"（fail 案例里没命中 evidence、原本
    # 整条丢弃的那批）已由教师直接候选级判定，落盘为
    # verifier/teacher_candidates_train.jsonl——并回训练池，让同一个 fail
    # 答案第一次同时含真正例和真负例候选，逼模型做候选级区分而不是学
    # "候选来自哪个答案"这个退化代理。
    teacher_path = root / "verifier" / "teacher_candidates_train.jsonl"
    teacher_labeled = load_teacher_candidate_labels(teacher_path) if teacher_path.exists() else []
    synthetic = synthetic_records_to_candidates(synthesize_all(train_records))
    n_positives = (
        sum(1 for it in trusted if it["label"])
        + sum(1 for it in teacher_labeled if it["label"])
        + len(synthetic)
    )
    # 结构性负例数量对齐正例总数（约1:1）——可信池里 pass 案例候选有
    # 5000+ 条，全装饰一遍会把正例占比从15.8%再腰斩到8.6%，重新逼近
    # ticket 09/11 坍缩过的区间；目的只是让"有没有括注"这个特征不再
    # 完美区分正负例，不需要每条负例都装饰一遍。
    structural_negatives = build_structural_negatives(
        train_records, jargon, holdout=False, max_items=n_positives
    )
    return trusted + teacher_labeled + synthetic + structural_negatives


def infer_is_positive(rows: list[dict]):
    """按数据集的输出 schema 自动选正例判定谓词——案例级看 `verdict==
    "fail"`，候选级（ticket 14）看 `violated` 布尔本身。

    **回归（code review 发现）**：`train()` 曾经不管数据是案例级还是候选级，
    永远用案例级的默认谓词（`verdict=="fail"`）算 `n_fail`/`n_fail_after`。
    候选级的输出 schema 是 `{"violated": bool}`，没有 `verdict` 字段，默认
    谓词对每一行都判 `None != "fail"` → 恒 False——过采样**静默变成
    no-op**，`n_fail`/`n_fail_after` 都打印 0，不报错，只是悄悄不起作用，
    正是这份训练脚本自己的 docstring 反复强调"不处理这一步会复现 09/11
    坍缩"的那类问题，没有任何信号能提前发现。改成按数据自动判定，不能
    让调用方凭记忆选对谓词。"""
    if not rows:
        raise ValueError("空数据集，无法判定正例谓词")
    sample = json.loads(rows[0]["output"])
    if "violated" in sample:
        return lambda o: bool(o["violated"])
    if "verdict" in sample:
        return lambda o: o.get("verdict") == "fail"
    raise ValueError(f"无法识别的输出 schema（既无 violated 也无 verdict 字段）：{sample}")


def train(data_path: Path, out_dir: Path, model: str, epochs: int, bsz: int,
          oversample_fail: int = 1) -> int:
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("需要 unsloth（只在 RunPod 上跑）：pip install unsloth", file=sys.stderr)
        return 1
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    rows = [json.loads(l) for l in data_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    is_positive = infer_is_positive(rows)
    n_fail = sum(1 for r in rows if is_positive(json.loads(r["output"])))
    print(f"训练样本 {len(rows)}（正例 {n_fail}, {n_fail / len(rows):.1%}）")
    if oversample_fail > 1:
        rows = oversample_positives(rows, oversample_fail, is_positive=is_positive)
        n_fail_after = sum(1 for r in rows if is_positive(json.loads(r["output"])))
        print(f"过采样 x{oversample_fail} 后 {len(rows)}（正例 {n_fail_after}, "
              f"{n_fail_after / len(rows):.1%}）")

    m, tok = FastLanguageModel.from_pretrained(
        model_name=model,
        max_seq_length=MAX_SEQ,
        # **不要开 load_in_4bit**：Unsloth 明确不建议对 Qwen3.5 做 4-bit QLoRA
        load_in_4bit=False,
        dtype=None,   # 让 Unsloth 自选 bf16
    )
    m = FastLanguageModel.get_peft_model(
        m,
        r=32,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=20260811,
    )

    def fmt(ex):
        return {"text": render_prompt(ex["system"], ex["input"]) + ex["output"] + tok.eos_token}

    ds = Dataset.from_list(rows).map(fmt)
    trainer = SFTTrainer(
        model=m,
        tokenizer=tok,
        train_dataset=ds,
        args=SFTConfig(
            per_device_train_batch_size=bsz,
            gradient_accumulation_steps=4,
            num_train_epochs=epochs,
            learning_rate=1e-4,
            warmup_ratio=0.05,
            logging_steps=5,
            optim="adamw_8bit",
            lr_scheduler_type="cosine",
            seed=20260811,
            output_dir=str(out_dir),
            report_to="none",
            dataset_text_field="text",
            max_seq_length=MAX_SEQ,
        ),
    )

    # **只在 assistant 段算 loss。** 不做这一步，prompt 里的原文+回答（p90 2594
    # 字）会把输出 JSON 里本就稀缺的红线信号（过采样前约 1.6%）进一步稀释——
    # 这是与过采样并列的第二道防线，两个都不做基本会复现 09 的坍缩。
    # instruction_part 是整段文本的起点（Base 模型模板没有更早的边界可用），
    # response_part 必须与 render_prompt 实际吐出的 assistant 段起始逐字节一致。
    from unsloth.chat_templates import train_on_responses_only

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|system|>\n",
        response_part=RESPONSE_MARKER,
    )

    trainer.train()
    m.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    print(f"已保存到 {out_dir}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-data", action="store_true", help="从标注产出训练集（本机跑）")
    ap.add_argument("--make-candidate-data", action="store_true",
                    help="组装候选级训练集（ticket 14，本机跑）：train 切分"
                         "可信池（pass案例负例+fail案例可信正例）+ train 切分"
                         "合成正例")
    ap.add_argument("--candidate-data-out", type=Path, default=Path("candidate_train.jsonl"))
    ap.add_argument("--data", type=Path, default=Path("train.jsonl"),
                    help="训练输入路径——train() 与 --make-candidate-data 都读它"
                         "（--make-data 例外：那个模式从 to_label.json/labels/*.jsonl"
                         "产出案例级数据，--data 只是它的写出路径）")
    ap.add_argument("--out", type=Path, default=Path("./ckpt"))
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bsz", type=int, default=2)
    ap.add_argument("--holdout", type=int, default=0,
                    help="留出多少条做验收（必须在训练前切出并封存）")
    ap.add_argument("--oversample-fail", type=int, default=1,
                    help="正例行重复几遍（案例级看 verdict=='fail'，候选级看"
                         "violated，train() 按输出 schema 自动判定）。ticket 11"
                         "要求：训练集红线正例曾低到约 1.6%%，不处理大概率复现 09"
                         "的坍缩。从 1（不采样）开始在 held-out 上网格搜索，"
                         "不要拍一个数")
    args = ap.parse_args()

    if args.make_candidate_data:
        # 惰性导入（同 build_candidate_training_pool 的理由：candidate_pool
        # 不在 RunPod 传输清单里，模块顶层 import 会在远程炸掉）——这里只
        # 要一个字符串常量，跟 build_candidate_training_pool 各自惰性导入
        # 一次，不算重复劳动。
        root = Path(__file__).resolve().parent.parent
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from verifier.candidate_pool import STRUCTURAL_NEGATIVE_SUFFIX

        # **读 args.data，不硬编码路径**（code review 发现：硬编码曾经让
        # --data 被静默忽略——同一个标志在这个脚本里既是 train() 的输入，
        # 也应该是这个模式的输入，不能各写一套）。
        train_records = [
            json.loads(line)
            for line in args.data.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        pool = build_candidate_training_pool(train_records)
        ds = make_candidate_dataset(pool)
        if not ds:
            print("候选级训练集为空——检查 jargon 词表/train 切分触发点是否正常，"
                  "不是继续往下跑的信号", file=sys.stderr)
            return 1
        n_pos = sum(1 for r in ds if json.loads(r["output"])["violated"])
        n_rl0 = sum(1 for r in pool if r["label"] and r.get("red_line_guess") == 0)
        n_rl2 = sum(1 for r in pool if r["label"] and r.get("red_line_guess") == 2)
        n_structneg = sum(1 for r in pool if r["case_id"].endswith(STRUCTURAL_NEGATIVE_SUFFIX))
        print(f"候选级训练集 {len(ds)} 条（正例 {n_pos}，{n_pos / len(ds):.1%}"
              f"——红线0 {n_rl0} / 红线2 {n_rl2}；负例 {len(ds) - n_pos}，"
              f"其中结构性负例 {n_structneg} 条——问题一的修复，打掉"
              "「括注即违规」这个纯结构捷径）")
        # 绝不只报总体准确率能算出的假象数字——这里只报正负比例，不是准确率，
        # 但仍然按项目纪律把红线0/2分开列，供后续训练配置参考。
        args.candidate_data_out.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in ds), encoding="utf-8"
        )
        print(f"-> {args.candidate_data_out}")
        return 0

    if args.make_data:
        root = Path(__file__).resolve().parent.parent
        items = json.loads((root / "verifier" / "work" / "to_label.json").read_text(encoding="utf-8"))
        labels = {}
        for p in sorted((root / "verifier" / "labels").glob("*.jsonl")):
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    j = json.loads(line)
                    labels[j["case_id"]] = j
        ds = make_dataset(items, labels)
        print(f"可用样本 {len(ds)}（标注 {len(labels)} 条）")

        if args.holdout:
            # **验收集必须在训练前切出并封存**，不能等训完再从剩余数据里挑——
            # 那等于自己给自己划及格线。切分按内容哈希，确定性。
            #
            # **按 fail/pass 分层切**，让验收集的正例比例略高于总体：正例是
            # 稀缺资源（440 条里只有 91 条 fail），随机切会让验收集正例更少、
            # 门槛更没法判。分层不改变"训练前封存"这一点。
            import hashlib

            def h(x):
                return int(hashlib.sha256(x["input"][:64].encode()).hexdigest()[:8], 16)

            fails = sorted([r for r in ds if '"verdict": "fail"' in r["output"]], key=h)
            passes = sorted([r for r in ds if '"verdict": "fail"' not in r["output"]], key=h)
            n_fail = min(len(fails), round(args.holdout * HOLDOUT_FAIL_RATIO))
            hold = fails[:n_fail] + passes[: args.holdout - n_fail]
            ds = fails[n_fail:] + passes[args.holdout - n_fail:]
            ds.sort(key=h)
            (root / "verifier" / "holdout.jsonl").write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in hold), encoding="utf-8"
            )
            print(f"验收集 {len(hold)} 条已封存 -> verifier/holdout.jsonl")

        args.data.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in ds), encoding="utf-8"
        )
        print(f"训练集 {len(ds)} 条 -> {args.data}")
        return 0

    return train(args.data, args.out, args.model, args.epochs, args.bsz, args.oversample_fail)


if __name__ == "__main__":
    sys.exit(main())
