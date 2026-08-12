"""verifier/gen_answers.py 的增量生成测试（ticket 10）。

扩量追加（build_teacher_labels.py --append）会把新样本接到已有 480 条后面，
其中大部分已经有 answer 了——不跳过重跑就会把 DeepSeek 预算在已完成的样本上
重花一遍。这里守的就是这条跳过逻辑。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import verifier.gen_answers as gen_answers  # noqa: E402


def test_skips_items_that_already_have_an_answer(monkeypatch, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    items = [
        {"case_id": "vt-0001", "name": "老药", "source_text": "已经标注过的源文本。",
         "answer": "已有的回答"},
        {"case_id": "vt-0002", "name": "新药", "source_text": "新追加的源文本。"},
    ]
    (work / "to_label.json").write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")

    monkeypatch.setattr(gen_answers, "WORK", work)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-key")

    called_ids = []

    def fake_call_model(c, model, key, base):
        called_ids.append(c["id"])
        return {"id": c["id"], "output": "生成的新回答", "finish_reason": "stop", "truncated": False}

    monkeypatch.setattr(gen_answers, "call_model", fake_call_model)

    assert gen_answers.main() == 0

    assert called_ids == ["vt-0002"], "只应该为没有 answer 的新样本调用模型"

    out = json.loads((work / "to_label.json").read_text(encoding="utf-8"))
    by_id = {c["case_id"]: c for c in out}
    assert by_id["vt-0001"]["answer"] == "已有的回答", "已有回答不能被覆盖"
    assert by_id["vt-0002"]["answer"] == "生成的新回答"


def test_all_answered_skips_every_call(monkeypatch, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    items = [{"case_id": "vt-0001", "name": "老药", "source_text": "文本", "answer": "已有"}]
    (work / "to_label.json").write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")

    monkeypatch.setattr(gen_answers, "WORK", work)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-key")

    def fail_if_called(*a, **k):
        raise AssertionError("不应该为已有回答的样本调用模型")

    monkeypatch.setattr(gen_answers, "call_model", fail_if_called)

    assert gen_answers.main() == 0
