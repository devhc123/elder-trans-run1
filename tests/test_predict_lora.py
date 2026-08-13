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

import pytest

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


# ---------- 候选级推理模式（ticket 18） ----------
#
# 候选级路径此前完全不存在：`predict_lora.py` 只 import 案例级 SYSTEM/build_prompt，
# `audit_prediction` 硬假设 key_points/red_lines/verdict，`load_items` 按 to_label.json
# 的 case_id 找条目——候选级 id（`vt-xxxx::候选文本`）会直接在缺失检查那里失败。
# 这一组守的是：prompt 与训练时字面一致、id 与训练时同一个函数产出、预测行不夹带 gold。

from verifier.predict_lora import (  # noqa: E402
    CANDIDATE_MAX_NEW_TOKENS,
    DEFAULT_MAX_NEW_TOKENS,
    audit_candidate_prediction,
    build_candidate_row,
    build_candidate_tasks,
    load_candidate_pool,
)
from verifier.train_lora import (  # noqa: E402
    SYSTEM_CANDIDATE,
    build_candidate_prompt,
    candidate_id,
    render_prompt,
)


def _pool_item(candidate_text="华法林", label=True):
    return {
        "case_id": "vt-0001",
        "source_text": "本品与抗凝类药物合用需注意。",
        "answer": "请遵医嘱。（补充一句：像华法林这类药，也是您问的这一类里常见的。）",
        "candidate_text": candidate_text,
        "kind": "lexicon",
        "red_line_guess": 0,
        "label": label,
    }


def test_candidate_task_prompt_is_byte_identical_to_training_format():
    """推理 prompt 与训练时**字面一致**是这个脚本的既有铁律（案例级那条
    docstring 写着"两份字面量模板不同步的偏差不会报错，只会让生成质量莫名
    下降"）——候选级必须走同样的复用，不能在这里重写模板。"""
    item = _pool_item()
    (row_id, prompt, aux), = build_candidate_tasks([item])
    assert prompt == render_prompt(SYSTEM_CANDIDATE, build_candidate_prompt(item))
    assert aux is item


def test_candidate_task_id_comes_from_the_single_definition():
    """id 必须由 `train_lora.candidate_id` 产出——训练集里的 case_id 就是它
    生成的，两边各写一份 f-string 会让验收时 join 不上，且不报错。"""
    item = _pool_item()
    (row_id, _, _), = build_candidate_tasks([item])
    assert row_id == candidate_id(item)


def test_audit_candidate_prediction_requires_a_boolean():
    assert audit_candidate_prediction({"violated": True}, _pool_item())["schema_ok"] is True
    assert audit_candidate_prediction({"violated": False}, _pool_item())["schema_ok"] is True
    # 字符串 "true"、缺字段、给了案例级 schema —— 都不算合格
    assert audit_candidate_prediction({"violated": "true"}, _pool_item())["schema_ok"] is False
    assert audit_candidate_prediction({}, _pool_item())["schema_ok"] is False
    assert audit_candidate_prediction({"verdict": "fail"}, _pool_item())["schema_ok"] is False


def test_audit_candidate_prediction_reports_zero_evidence():
    """候选级 schema 没有 evidence 字段。统计口径要与案例级共用同一个累加器，
    所以必须返回 0 而不是缺键——缺键会在累加时 KeyError。"""
    diag = audit_candidate_prediction({"violated": True}, _pool_item())
    assert diag["evidence_total"] == 0 and diag["evidence_grounded"] == 0


def test_candidate_prediction_row_does_not_carry_gold():
    """预测文件里不许出现 gold 标签——验收脚本从 pool 取 gold，预测自带答案
    等于给自己留了一条"验收时不小心读到答案"的后门。"""
    row = build_candidate_row("vt-0001::华法林", {"violated": True})
    assert row == {"case_id": "vt-0001::华法林", "violated": True}


def test_candidate_row_keeps_malformed_violated_visible():
    """schema 不合格时也要落盘（否则这条就从验收集里凭空消失了），但要让
    验收侧看得出它不是一个合法的 True。"""
    row = build_candidate_row("x", {"violated": "true"})
    assert row["case_id"] == "x"
    assert row["violated"] is not True


def test_candidate_generation_budget_is_far_smaller_than_case_level():
    """候选级目标就是 `{"violated": true}` 十几个 token。沿用案例级的 768
    会让 2,100+ 次串行生成白白拖长机时——按 ticket 09 实测的速度，这是
    真金白银的 RunPod 机时，不是纸面优化。"""
    assert CANDIDATE_MAX_NEW_TOKENS < DEFAULT_MAX_NEW_TOKENS / 10


def test_load_candidate_pool_reads_json_list(tmp_path):
    p = tmp_path / "pool.json"
    p.write_text(json.dumps([_pool_item()], ensure_ascii=False), encoding="utf-8")
    assert load_candidate_pool(p)[0]["candidate_text"] == "华法林"


def test_candidate_mode_never_touches_to_label_json(tmp_path, monkeypatch):
    """候选 track 的 scp 清单里**没有** `to_label.json`——远程跑推理时那个
    文件根本不存在。`main()` 现在无条件调 `load_items()`，候选级模式必须
    绕开它，否则一上机就 FileNotFoundError。"""
    import verifier.predict_lora as pl

    pool_path = tmp_path / "pool.json"
    pool_path.write_text(json.dumps([_pool_item()], ensure_ascii=False), encoding="utf-8")

    def _boom():
        raise AssertionError("候选级模式不该读 to_label.json")

    captured = {}

    def _fake_run(ckpt, tasks, out_path, max_new_tokens, checkpoint_every, audit, make_row):
        captured["tasks"] = tasks
        captured["max_new_tokens"] = max_new_tokens
        return {"n": 0, "parse_fail": 0, "schema_fail": 0,
                "evidence_total": 0, "evidence_grounded": 0}

    monkeypatch.setattr(pl, "load_items", _boom)
    monkeypatch.setattr(pl, "run_tasks", _fake_run)
    monkeypatch.setattr(
        sys, "argv",
        ["predict_lora.py", "--ckpt", str(tmp_path), "--candidates-from", str(pool_path),
         "--out", str(tmp_path / "pred.jsonl")],
    )
    assert pl.main() == 0
    assert captured["tasks"][0][0] == "vt-0001::华法林"
    assert captured["max_new_tokens"] == CANDIDATE_MAX_NEW_TOKENS


# ---------- DeepSeek 独立审计（ticket 21）发现并已复现的五处 ----------

def test_audit_prediction_survives_wrong_container_types():
    """**DeepSeek #1（SEVERE）**：生成式模型完全可能吐 `{"key_points": {...}}`
    （dict 不是 list）或 `{"key_points": ["x"]}`（元素不是 dict）。旧写法会在
    `(kp or []) + (rl or [])` 抛 TypeError、或在 `k.get` 抛 AttributeError，
    **整轮推理当场中断**——最后一个 checkpoint 之后的结果全丢，而这是在按秒
    计费的 GPU 上。形状异常必须记成 schema 违规，不是崩溃。"""
    for bad in ({"key_points": {"covered": True}, "red_lines": [], "verdict": "pass"},
                {"key_points": ["x"], "red_lines": [], "verdict": "pass"},
                {"key_points": [{"covered": True, "evidence": ["不是字符串"]}],
                 "red_lines": [], "verdict": "pass"}):
        diag = audit_prediction(bad, _gold_item())
        assert diag["schema_ok"] is False          # 记成违规
        assert isinstance(diag["evidence_total"], int)   # 而且没炸


def test_empty_task_list_fails_before_loading_the_model():
    """**DeepSeek #2**：空任务会加载完模型跑一个空循环、不写输出、还以 0 退出
    ——白烧一次建机+装依赖+载模型的机时，下游却找不到 pred 文件。"""
    import verifier.predict_lora as pl
    with pytest.raises(SystemExit, match="空"):
        pl.run_tasks(Path("/tmp/x"), [], Path("/tmp/o.jsonl"), 16, 10, lambda p, a: {}, lambda i, p: {})


def test_nonpositive_checkpoint_every_is_rejected():
    """**DeepSeek #5**：`i % 0` 会 ZeroDivisionError，同样是在 GPU 上才炸。"""
    import verifier.predict_lora as pl
    tasks = [("id", "prompt", {})]
    with pytest.raises(SystemExit, match="checkpoint-every"):
        pl.run_tasks(Path("/tmp/x"), tasks, Path("/tmp/o.jsonl"), 16, 0, lambda p, a: {}, lambda i, p: {})


def test_candidate_pool_must_be_a_json_array(tmp_path):
    """**DeepSeek #7**：顶层是 dict 时会去遍历它的键（字符串），产出一堆无意义
    prompt 或在 candidate_id 里炸——都是上了 GPU 才发现。"""
    p = tmp_path / "pool.json"
    p.write_text(json.dumps({"items": [_pool_item()]}, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="不是 JSON 数组"):
        load_candidate_pool(p)


def test_two_modes_are_mutually_exclusive_at_the_arg_layer(tmp_path, monkeypatch):
    """**DeepSeek #3**：两个都传时，候选级模式会静默忽略案例级那个，跑完才
    发现只出了一半结果。交给 argparse 强制。"""
    import verifier.predict_lora as pl
    monkeypatch.setattr(
        sys, "argv",
        ["predict_lora.py", "--ckpt", str(tmp_path),
         "--case-ids-from", str(tmp_path / "h.jsonl"),
         "--candidates-from", str(tmp_path / "p.json")],
    )
    with pytest.raises(SystemExit):
        pl.main()
