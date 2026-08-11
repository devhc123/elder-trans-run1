"""LoRA 训练脚本的测试（ticket 09）。

训练本身跑在 RunPod，本机测不了。所以这里守的是**配置约束**——那些错了
不会报错、只会让训练白跑或结果失真的地方。每一条都对应一份核实过的依据。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from verifier.train_lora import (  # noqa: E402
    BASE_MODEL,
    FALLBACK_MODEL,
    MAX_SEQ,
    SCALE_UP_MODEL,
    SYSTEM,
    build_prompt,
    build_target,
    make_dataset,
)


# ---------- 基座选型（依据见 docs/DATASETS_AND_BENCHMARKS.md §4） ----------

def test_uses_base_not_instruct():
    """必须用 Base：判官输出是固定 JSON schema，不需要对话能力；
    且 Unsloth 官方警告 Qwen3.5-2B **指令版**更易陷入 thinking 死循环，
    批量跑时是不终止生成的灾难。"""
    for m in (BASE_MODEL, FALLBACK_MODEL, SCALE_UP_MODEL):
        assert m.endswith("-Base"), m


def test_all_candidate_models_are_apache_licensed_families():
    """Qwen2.5-3B 是 qwen-research 许可证（限制商用），不能混进来。"""
    for m in (BASE_MODEL, FALLBACK_MODEL, SCALE_UP_MODEL):
        assert "Qwen2.5" not in m, f"{m} 可能落在 qwen-research 许可证下"


def test_scale_up_path_is_same_architecture():
    """未达门槛时的升级路径必须同架构同工具链，训练脚本零改动。"""
    assert SCALE_UP_MODEL.startswith("Qwen/Qwen3.5")
    assert BASE_MODEL.startswith("Qwen/Qwen3.5")


# ---------- prompt / target ----------

def _case():
    return {
        "case_id": "vt-0001",
        "source_text": "本品与抗血小板类药物合用时需注意出血风险。",
        "answer": "您这个药和阿司匹林一起吃要小心出血。",
        "key_points": ["与抗血小板类药物合用需注意出血风险"],
        "red_lines": ["把类别替换成具体药名", "把原文事实说反"],
    }


def _label():
    return {
        "case_id": "vt-0001",
        "key_points": [{"idx": 0, "covered": True, "evidence": "一起吃要小心出血"}],
        "red_lines": [
            {"idx": 0, "violated": True, "evidence": "阿司匹林"},
            {"idx": 1, "violated": False, "evidence": ""},
        ],
        "verdict": "fail",
    }


def test_prompt_contains_all_four_inputs():
    p = build_prompt(_case())
    for part in ("原文", "回答", "要点", "红线"):
        assert part in p
    assert _case()["source_text"] in p and _case()["answer"] in p


def test_prompt_numbers_items_so_indices_line_up():
    """要点/红线必须带下标——模型输出的 idx 要能和输入对上。"""
    p = build_prompt(_case())
    assert "0. " in p


def test_target_is_single_line_json():
    t = build_target(_label())
    assert "\n" not in t
    parsed = json.loads(t)
    assert parsed["verdict"] == "fail"


def test_target_keeps_evidence():
    """evidence 是可程序化校验的防幻觉抓手，也是汇报凭据，不能省。"""
    parsed = json.loads(build_target(_label()))
    assert parsed["red_lines"][0]["evidence"] == "阿司匹林"


def test_target_preserves_item_counts():
    parsed = json.loads(build_target(_label()))
    assert len(parsed["key_points"]) == 1
    assert len(parsed["red_lines"]) == 2


# ---------- system prompt ----------

def test_system_states_the_source_only_discipline():
    """「医学上对但原文没有 = 仍算违规」是本 verifier 的定义性纪律，
    必须写进 system，否则学出来的是通用医学问答判别器。"""
    assert "只以" in SYSTEM and "原文" in SYSTEM
    assert "医学上正确" in SYSTEM or "医学上对" in SYSTEM


def test_system_forbids_free_form_explanation():
    assert "只输出 JSON" in SYSTEM or "不要解释" in SYSTEM


# ---------- 数据集构造 ----------

def test_make_dataset_skips_unlabeled_items():
    ds = make_dataset([_case(), {**_case(), "case_id": "vt-9999"}],
                      {"vt-0001": _label()})
    assert len(ds) == 1


def test_make_dataset_rows_are_training_ready():
    ds = make_dataset([_case()], {"vt-0001": _label()})
    r = ds[0]
    assert set(r) == {"system", "input", "output"}
    assert all(isinstance(v, str) and v for v in r.values())


# ---------- 序列长度 ----------

def test_max_seq_covers_the_actual_data():
    """MAX_SEQ 太小会静默截断样本，训出来的模型看不到红线部分。"""
    work = ROOT / "verifier" / "work" / "to_label.json"
    if not work.exists():
        pytest.skip("待标注包未生成")
    items = json.loads(work.read_text(encoding="utf-8"))
    lens = sorted(len(build_prompt(c)) for c in items if c.get("answer"))
    if not lens:
        pytest.skip("尚无回答")
    p90 = lens[int(len(lens) * 0.9)]
    # 中文约 1 字 ≈ 1 token 的保守估计，留一倍余量给输出
    assert p90 * 2 < MAX_SEQ, f"P90 prompt {p90} 字，MAX_SEQ={MAX_SEQ} 可能不够"
