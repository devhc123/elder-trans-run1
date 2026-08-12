"""verifier 推理脚本的测试（ticket 09 摸底训练的下半场）。

推理本身要跑在 RunPod（需要 unsloth + 训好的 LoRA 权重），本机测不了。
这里守的是**解析与判分逻辑**——模型输出不是合法 JSON、evidence 是编造的、
schema 条数对不上，这些错了不会在生成阶段报错，只会让「2B 离教师差多少」
这个数字算错，而这正是 ticket 09 唯一要回答的问题。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from verifier.predict_lora import (  # noqa: E402
    audit_prediction,
    evidence_is_grounded,
    load_case_ids,
    parse_prediction,
)


# ---------- parse_prediction：模型输出不保证干净 ----------

def test_parse_prediction_plain_json():
    raw = '{"key_points": [], "red_lines": [], "verdict": "pass"}'
    assert parse_prediction(raw) == json.loads(raw)


def test_parse_prediction_strips_markdown_fence():
    raw = '```json\n{"key_points": [], "red_lines": [], "verdict": "pass"}\n```'
    assert parse_prediction(raw)["verdict"] == "pass"


def test_parse_prediction_ignores_leading_and_trailing_prose():
    """训练目标是纯 JSON，但没训好的 checkpoint 可能夹带解释——
    摸底阶段本来就是要测这个「schema 稳不稳」。"""
    raw = '好的，我来判断一下。\n{"key_points": [], "red_lines": [], "verdict": "fail"}\n以上是我的判断。'
    parsed = parse_prediction(raw)
    assert parsed is not None and parsed["verdict"] == "fail"


def test_parse_prediction_returns_none_on_garbage():
    assert parse_prediction("这不是 JSON，模型崩了") is None


def test_parse_prediction_returns_none_on_truncated_json():
    """MAX_SEQ/max_new_tokens 不够时输出会被截断——截断的 JSON 不该被当成
    一个"碰巧字段缺失"的合法预测，必须能和真正的合法输出区分开。"""
    assert parse_prediction('{"key_points": [{"idx": 0, "covered": tr') is None


def test_parse_prediction_returns_none_on_empty_string():
    assert parse_prediction("") is None


# ---------- evidence_is_grounded：program化校验，防幻觉 ----------

def _item():
    return {
        "case_id": "vt-0001",
        "source_text": "本品与抗血小板类药物合用时需注意出血风险。",
        "answer": "您这个药和阿司匹林一起吃要小心出血。",
    }


def test_evidence_grounded_in_source():
    assert evidence_is_grounded("抗血小板类药物", _item())


def test_evidence_grounded_in_answer():
    assert evidence_is_grounded("阿司匹林", _item())


def test_evidence_empty_string_is_trivially_grounded():
    """空 evidence 表示"没有证据可给"（例如 covered=false 时），不是编造，
    和 build_teacher_labels.py 的校验规则保持一致：`if ev and ev not in ...`。"""
    assert evidence_is_grounded("", _item())


def test_evidence_not_grounded_when_hallucinated():
    assert not evidence_is_grounded("华法林", _item())


# ---------- audit_prediction：schema 完整性 + evidence 合规率 ----------

def _gold_item():
    return {
        "case_id": "vt-0001",
        "source_text": "本品与抗血小板类药物合用时需注意出血风险。",
        "answer": "您这个药和阿司匹林一起吃要小心出血。",
        "key_points": ["与抗血小板类药物合用需注意出血风险"],
        "red_lines": ["把类别替换成具体药名", "把原文事实说反", "编造具体数字"],
    }


def test_audit_prediction_schema_ok_when_counts_match():
    pred = {
        "key_points": [{"idx": 0, "covered": True, "evidence": "阿司匹林一起吃要小心出血"}],
        "red_lines": [
            {"idx": 0, "violated": True, "evidence": "阿司匹林"},
            {"idx": 1, "violated": False, "evidence": ""},
            {"idx": 2, "violated": False, "evidence": ""},
        ],
        "verdict": "fail",
    }
    diag = audit_prediction(pred, _gold_item())
    assert diag["schema_ok"] is True


def test_audit_prediction_schema_fails_on_count_mismatch():
    """条数对不上是训练/推理最常见的失稳模式——prompt 给了 3 条红线，
    模型只答了 2 条，不能被当成"部分正确"，要显式记成 schema 违规。"""
    pred = {
        "key_points": [{"idx": 0, "covered": True, "evidence": ""}],
        "red_lines": [{"idx": 0, "violated": False, "evidence": ""}],
        "verdict": "pass",
    }
    diag = audit_prediction(pred, _gold_item())
    assert diag["schema_ok"] is False


def test_audit_prediction_counts_evidence_compliance():
    pred = {
        "key_points": [{"idx": 0, "covered": True, "evidence": "阿司匹林一起吃要小心出血"}],
        "red_lines": [
            {"idx": 0, "violated": True, "evidence": "华法林"},   # 编造，不合规
            {"idx": 1, "violated": False, "evidence": ""},        # 空，合规
            {"idx": 2, "violated": False, "evidence": ""},
        ],
        "verdict": "fail",
    }
    diag = audit_prediction(pred, _gold_item())
    assert diag["evidence_total"] == 2      # 两条非空 evidence
    assert diag["evidence_grounded"] == 1   # 只有一条是原文/回答子串


def test_audit_prediction_missing_verdict_is_schema_fail():
    pred = {
        "key_points": [{"idx": 0, "covered": True, "evidence": ""}],
        "red_lines": [
            {"idx": 0, "violated": False, "evidence": ""},
            {"idx": 1, "violated": False, "evidence": ""},
            {"idx": 2, "violated": False, "evidence": ""},
        ],
    }
    diag = audit_prediction(pred, _gold_item())
    assert diag["schema_ok"] is False


# ---------- load_case_ids ----------

def test_load_case_ids_reads_from_jsonl(tmp_path):
    p = tmp_path / "holdout.jsonl"
    p.write_text(
        '{"case_id": "vt-0001", "system": "s", "input": "i", "output": "o"}\n'
        '{"case_id": "vt-0002", "system": "s", "input": "i", "output": "o"}\n',
        encoding="utf-8",
    )
    assert load_case_ids(p) == ["vt-0001", "vt-0002"]


def test_load_case_ids_skips_blank_lines(tmp_path):
    p = tmp_path / "holdout.jsonl"
    p.write_text(
        '{"case_id": "vt-0001", "system": "s", "input": "i", "output": "o"}\n\n'
        '{"case_id": "vt-0002", "system": "s", "input": "i", "output": "o"}\n',
        encoding="utf-8",
    )
    assert load_case_ids(p) == ["vt-0001", "vt-0002"]
