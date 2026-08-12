"""教师标注抽样的测试（ticket 10）。

09 摸底证伪了原始 SAMPLE_MIX 的假设（thin_source/negation/conditional 均不如
plain 对照臂），并发现 verifier 会在正例密度不够时坍缩成常量输出。这里守的是
按此重配后的配比，以及**扩量必须能安全追加**这条——record_id 相同的样本决不能
在扩量时换发新 case_id，否则会冲掉 verifier/labels/ 里已经标注好的 440 条。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from verifier.build_teacher_labels import (  # noqa: E402
    SAMPLE_MIX,
    assemble,
    sample,
)


# ---------- 重配后的配比（依据：ticket 09 的 440 条实测 fail 率） ----------

def test_sample_mix_sums_to_one():
    assert abs(sum(SAMPLE_MIX.values()) - 1.0) < 1e-9


def test_category_is_the_only_bucket_boosted_above_plain():
    """09 实测：category 27.2% 是唯一跑赢 plain 对照臂（26.7%）的难例桶，
    其余三桶（18.1%/17.1%/12.0%）全部不如随机抽样——继续重仓它们没有依据。"""
    assert SAMPLE_MIX["category"] > SAMPLE_MIX["plain"]
    for weak in ("thin_source", "negation", "conditional"):
        assert SAMPLE_MIX[weak] < SAMPLE_MIX["category"]


def test_deprioritized_buckets_combined_are_a_minority():
    """三个证伪的难例桶合计权重必须低于原配比（原是 0.25+0.20+0.15=0.60）。"""
    weak_total = sum(SAMPLE_MIX[b] for b in ("thin_source", "negation", "conditional"))
    assert weak_total < 0.35


def test_plain_control_arm_still_present():
    """对照臂不能被砍到 0——没有它就再也读不出任何难例采样是否有效。"""
    assert SAMPLE_MIX["plain"] >= 0.20


# ---------- sample() 的 exclude 参数（扩量安全追加的基础） ----------

def _buckets():
    return {
        "category": [{"record_id": f"cat-{i}"} for i in range(10)],
        "plain": [{"record_id": f"pln-{i}"} for i in range(10)],
        "thin_source": [{"record_id": f"thn-{i}"} for i in range(10)],
        "negation": [{"record_id": f"neg-{i}"} for i in range(10)],
        "conditional": [{"record_id": f"cnd-{i}"} for i in range(10)],
    }


def test_sample_excludes_already_picked_record_ids():
    b = _buckets()
    first = sample(b, 20)
    picked_ids = {c["record_id"] for c in first}
    second = sample(b, 20, exclude=picked_ids)
    assert not ({c["record_id"] for c in second} & picked_ids)


def test_sample_without_exclude_is_deterministic_prefix():
    """不传 exclude 时，同一路数字的抽样结果必须是同一份排序的前缀——
    这是「原始 --sample 结果是扩量时可复现的前缀」这条不变量的基础。"""
    b = _buckets()
    small = sample(b, 10)
    big = sample(b, 20)
    small_ids = {c["record_id"] for c in small}
    big_ids = {c["record_id"] for c in big}
    assert small_ids <= big_ids


# ---------- assemble()：case_id 编号与追加 ----------

def _case(rid):
    return {
        "record_id": rid,
        "source_text": "本品为口服抗凝药，成人常用量一日一次。" "服药期间需定期复查凝血功能与肝肾功能。",
    }


def test_assemble_assigns_sequential_case_ids_from_start():
    out = assemble([_case("a"), _case("b")], start=1)
    assert [c["case_id"] for c in out] == ["vt-0001", "vt-0002"]


def test_assemble_continues_numbering_from_existing_max():
    """扩量追加时新样本的 case_id 必须接着已有最大编号走，不能从 1 重排——
    重排会让 verifier/labels/ 里按旧 case_id 存的教师标注全部对不上号。"""
    out = assemble([_case("c")], start=501)
    assert out[0]["case_id"] == "vt-0501"


def test_assemble_filters_short_key_points():
    tiny = {"record_id": "z", "source_text": "太短。"}
    out = assemble([tiny], start=1)
    assert out == []
