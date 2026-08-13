#!/usr/bin/env python3
"""退化分类器探针：把"捷径"从未知漏洞变成预注册的对照臂（ticket 17）。

**这个模块解决的是一个流程问题，不是一个算法问题。** ticket 14 的历史是：
第三轮审计发现"只看答案里有没有固定括注模板"就能在对抗子集上拿 100% 召回 /
0% 误报 → 修（零候选安慰语）→ 下一轮审计又发现"只看候选是不是落在括注之后"
照样 100%/0.4%。每一轮都是人工侦查发现一个新特征，每一轮都在花掉一轮审计。

这里把那些侦查动作固化成**每次重建数据都自动跑**的一组退化分类器——它们
完全不读 `source_text`、不做任何语义判断，只看结构。如果其中任何一个能把
正负例分开，那说明数据集里有一条不需要理解就能走的捷径。

## 证据是单向的（L5，Feng, Wallace & Boyd-Graber, ACL 2019, arXiv:1905.05778）

这个模块有 8 年的文献谱系（**partial-input baseline / hypothesis-only baseline**，
Gururangan et al. NAACL 2018 arXiv:1803.02324：只看假设不看前提就能对 SNLI 的 67%）。
而那条谱系里最要紧的一条结论是**证据不对称**：

> 探针**成功**说明这份数据能被作弊；
> 探针**失败不能说明这份数据没有伪影**——它可能只是掩盖了"只在完整输入里才看得见"的
> 更平凡的模式。

所以本模块的门槛**只在一个方向上有效力**：超线即否决；**未超线只是"没被证伪"，
不构成"这份池干净"的任何正面证据**。代码与报告里都不许出现"通过=干净"的说法。

同一篇论文也给了正面指引：单视角探针不够，必须有**同时读两侧输入、但只做平凡处理**的
探针（他们实测：hypothesis-only 模型叠加前提里的平凡模式，能解掉此前被判为 "hard" 的
样本中的 15%）。`conjunction_in_paren_and_not_grounded` 就是这一条，**它永不从套件里
移除**，即使它按构造就接近满分、即使它只报告不设门。

## 三档门槛（预注册，不许测完再调）

- **纯结构探针（`STRUCTURE_PROBES`）**：只在**对抗子集**上设硬门
  `STRUCTURE_GATE_J`，超过即判数据集缺陷。这四个特征与内容完全无关，
  冻结验收集上出现任何区分力都是缺陷；构造上应当是 0，0.3 只是留给数据
  漂移的余量。
- **内容型探针（`REPORT_ONLY_PROBES`）**：两个池上都**只报告不设门**。
  「正例 = 注入的非 grounded 项」这个等式**就是数据构造本身**——只要正例
  靠规则注入产生、且注入内容 grounded 与否完全决定标签，"识别注入 ∧ 字符串
  匹配"的合取就永远接近满分。这不是可以靠再加一批负例修掉的漏洞，是规则
  合成数据集的固有性质（FactCC 谱系同样如此）。**承认它比假装堵上了诚实**：
  唯一的硬判决因此放在真实 holdout 池上（那里根本没有注入结构），
  合成池与真实池漏报率差 >10pp 判"学到伪影"（L1）。
- **训练池**：所有探针都只报告。训练池结构探针的 J 上限恒等于合成正例占比
  ——要把它压到 0.3 得把注入负例堆到数千条，代价是正例占比逼近 ticket 09/11
  的坍缩区间，换来的全是"字符串查表就能做对"的便宜负例。正确的解法是加
  真实正例（ticket 23），不是加便宜负例。

用法：
    python3 verifier/shortcut_probes.py --pool verifier/work/adversarial_subset_holdout.json --gate
    python3 verifier/shortcut_probes.py --all      # 三个池的完整基线表
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from verifier.redline_candidates import (  # noqa: E402
    CN_NUMERAL_RE,
    DIGIT_RE,
    FRACTION_RE,
    normalize,
)
from verifier.synth_minimal_edit import TEMPLATES, TEMPLATE_PREFIXES  # noqa: E402

WORK = ROOT / "verifier" / "work"

# 纯结构探针在**冻结验收集**上的 Youden J 上限。构造上应当是 0。
STRUCTURE_GATE_J = 0.3

# 注入括注的收尾。`probe_answer_ends_with_paren` 原来把它硬编码成字面量，与
# 生成模板是**隐式**耦合——模板一改措辞，探针就静默失效，而探针失效的表现是
# J=0（"很干净"），方向最坏（DeepSeek 独立审计 #4）。这里做成常量 + 下面的
# 断言，把耦合显式化：改坏了立刻炸，不是悄悄放行。
TEMPLATE_SUFFIX = "。）"
assert all(t.endswith(TEMPLATE_SUFFIX) for t in TEMPLATES), (
    "TEMPLATES 改了收尾但 TEMPLATE_SUFFIX 没同步——"
    "probe_answer_ends_with_paren 会静默失效，而失效看起来像「没有捷径」"
)

# "候选落在答案末尾多少比例内"算尾部。第三轮审计实测：合成候选 100% 落在
# 答案最后 15%，真实正例候选只有 7.4% 落在这个区间。
TAIL_FRACTION = 0.15


def infer_kind(text: str) -> str:
    """按候选文本本身推断 `extract_candidates` 会给它打的 kind。

    **为什么需要推断**：`synthetic_records_to_candidates` 产出的条目只有
    `red_line_guess`，**没有 `kind` 字段**，而对抗子集的正例全部来自那里。
    如果探针因此把数字候选按词表词的口径判 grounded（子串 vs 整段匹配），
    正负例两边用的就不是同一把尺子，测出来的 J 是假的——比没有测量更糟。
    """
    if FRACTION_RE.fullmatch(text):
        return "fraction"
    if DIGIT_RE.fullmatch(text):
        return "digit"
    if CN_NUMERAL_RE.fullmatch(text):
        return "cn_numeral"
    return "lexicon"


def _assert_kind_matches_shape(pool: list[dict]) -> None:
    """池里带 `kind` 字段的条目，字段必须与候选文本的实际形状一致。

    **这条断言是踩出来的。** 修 digit 捷径时，注入负例的 `candidate_text` 写成了
    「3天」，而段A 的 `DIGIT_RE` 从答案里抽出来的是裸数字「3」；存储的 kind 写着
    `digit`、`infer_kind("3天")` 推断出的是 `lexicon`。探针用字段、临时核查脚本
    用推断，两边给出不同的数字，**都不报错**——表现成"修复看起来没生效"，
    真正的问题却是候选形状与生产分布对不上（A3 那轮修过的同类错误）。

    实测：三个生产池 7,000+ 条带 kind 的条目，当前不一致 **0** 条。所以这是在
    固化一条当下成立的不变量，不是许愿。没有 `kind` 字段的条目（合成正例走的
    那条路径）不在此列，它们本来就靠推断。"""
    bad = [it for it in pool
           if "kind" in it and it["kind"] != infer_kind(it["candidate_text"])]
    if bad:
        sample = [(it["candidate_text"], it["kind"], infer_kind(it["candidate_text"]))
                  for it in bad[:5]]
        raise ValueError(
            f"{len(bad)} 条候选的 kind 字段与文本形状不符（候选, 字段, 推断）：{sample}。"
            f"多半是候选文本写成了段A 不会产出的形状——比如注入「3天」但段A 抽的是"
            f"裸数字「3」。这会让红线2 的验收测一个生产环境不存在的分布。"
        )


def _parenthetical_start(answer: str) -> int:
    """答案里最早出现的注入括注前缀的位置；没有返回 -1。"""
    idxs = [i for i in (answer.find(p) for p in TEMPLATE_PREFIXES) if i != -1]
    return min(idxs) if idxs else -1


def probe_template_prefix_present(item: dict) -> bool:
    """答案里有没有注入用的括注前缀。第三轮审计发现的原始捷径。"""
    return _parenthetical_start(item["answer"]) != -1


def probe_candidate_in_parenthetical(item: dict) -> bool:
    """候选文本是不是出现在括注**里面**。**问题一的修复没堵住的那个变种**：
    零候选安慰语让"括注在不在"失效了，但安慰语里刻意零候选，于是"候选在不在
    括注里"照样把两类完全分开。

    **必须卡住右括号**（DeepSeek 独立审计 #3）：只判"在前缀之后"会把括注**外面**
    但位置靠后的候选也算进来（实测复现：`（补充一句：…）随后予布洛芬` 里的
    "布洛芬"被判成在括注内）。测量仪器本身失准，量出来的 J 就是错的。"""
    answer = item["answer"]
    cut = _parenthetical_start(answer)
    if cut == -1:
        return False
    close = answer.find("）", cut)
    end = len(answer) if close == -1 else close
    return answer.find(item["candidate_text"], cut, end) != -1


def probe_candidate_in_last_15pct(item: dict) -> bool:
    """候选**有没有一次出现**落在答案尾部——不依赖任何模板措辞的位置特征，
    换一套括注写法也躲不掉它。

    用最后一次出现而不是第一次（DeepSeek 独立审计 #5）：候选在正文里自然出现过
    一次、又被注入句用了一次时，取首次出现会判"不在尾部"，**低估**尾部捷径。
    探针问的是"存不存在一次尾部出现"，那就该看最后一次。"""
    answer = item["answer"]
    idx = answer.rfind(item["candidate_text"])
    return idx != -1 and idx >= (1 - TAIL_FRACTION) * len(answer)


def probe_answer_ends_with_paren(item: dict) -> bool:
    """答案是不是以注入括注的收尾结束——注入句永远追加在末尾，真实答案几乎
    不命中。收尾从 `TEMPLATE_SUFFIX` 取，与生成模板的耦合由模块顶部的断言守。"""
    return item["answer"].rstrip().endswith(TEMPLATE_SUFFIX)


def probe_candidate_not_grounded(item: dict) -> bool:
    """候选在原文里找不到。**口径必须与 `extract_candidates` 完全一致**：
    数字/中文数词/分数整段匹配（"12" 是 "112" 的子串但 12≠112），词表词
    沿用朴素子串（中文复合词的子串通常仍是同一实体）。

    这条是内容型探针：它其实就是段A的 grounding 过滤器本身。生产环境里
    段A**只会**吐出非 grounded 的候选，所以一个靠它决策的模型部署后等价于
    恒判违规——正因如此它上不了门槛，但必须报出来。
    """
    text = normalize(item["candidate_text"])
    nsrc = normalize(item["source_text"])
    kind = item.get("kind") or infer_kind(item["candidate_text"])
    if kind == "lexicon":
        return text not in nsrc
    spans = {
        "digit": DIGIT_RE,
        "cn_numeral": CN_NUMERAL_RE,
        "fraction": FRACTION_RE,
    }[kind].findall(nsrc)
    return text not in set(spans)


def probe_conjunction_in_paren_and_not_grounded(item: dict) -> bool:
    """"在括注内 ∧ 不在原文里"。**这一条是规则合成集的天花板**：正例恰好
    就是"注入进去的、原文没有的那个词"，所以这个合取在任何配比下都接近满分。
    加 grounded 注入负例杀第一项、加等价形式注入负例杀第二项，但两项同时
    为假的那一格（在括注内、且原文没有、却又合法）按红线定义**不存在**
    ——除非教师标注。所以它只报告，不设门。"""
    return probe_candidate_in_parenthetical(item) and probe_candidate_not_grounded(item)


def probe_candidate_kind_is_lexicon(item: dict) -> bool:
    """候选是不是词表实体。不是结构捷径，是**分布探针**：等价形式负例全是
    数字类，大量加入会让 kind 分布在正负例间倾斜，于是"是不是药名"本身
    变成一个可学信号。这条的用法是跟基线比趋势，不是看绝对值。"""
    kind = item.get("kind") or infer_kind(item["candidate_text"])
    return kind == "lexicon"


def probe_candidate_is_digit(item: dict) -> bool:
    """候选是不是阿拉伯数字形。**Fable 5 审计 B 的直接产物。**

    上一版套件只有 `candidate_kind_is_lexicon`（词表词 vs 其他），对
    digit / cn_numeral 之间的倾斜完全瞎。而实测那正是当时最强的一条活口：
    等价形式负例的方向受原文写法支配（说明书几乎全用阿拉伯数字 → 产出的多是
    中文数词形），数字类**正例**的候选却多是阿拉伯数字形——于是"括注内是不是
    digit"在红线2 切片上拿到 **J = 0.722**，完全不用读原文。

    加它是为了让同一个疏忽下次立刻显形：kind 是候选文本的形状，模型不需要
    任何理解就能看见。"""
    kind = item.get("kind") or infer_kind(item["candidate_text"])
    return kind == "digit"


PROBES: tuple[tuple[str, object], ...] = (
    ("template_prefix_present", probe_template_prefix_present),
    ("candidate_in_parenthetical", probe_candidate_in_parenthetical),
    ("candidate_in_last_15pct", probe_candidate_in_last_15pct),
    ("answer_ends_with_paren", probe_answer_ends_with_paren),
    ("candidate_not_grounded", probe_candidate_not_grounded),
    ("conjunction_in_paren_and_not_grounded", probe_conjunction_in_paren_and_not_grounded),
    ("candidate_kind_is_lexicon", probe_candidate_kind_is_lexicon),
    ("candidate_is_digit", probe_candidate_is_digit),
)

STRUCTURE_PROBES = (
    "template_prefix_present",
    "candidate_in_parenthetical",
    "candidate_in_last_15pct",
    "answer_ends_with_paren",
)

REPORT_ONLY_PROBES = (
    "candidate_not_grounded",
    "conjunction_in_paren_and_not_grounded",
    "candidate_kind_is_lexicon",
    "candidate_is_digit",
)

# **同时读两侧输入的探针，永不移除**（L5 / Feng et al. arXiv:1905.05778）。
# 它按构造就接近满分、又只报告不设门，看起来像"没用、可以删掉"——恰恰相反：
# 单视角探针漏掉的伪影只有这种联合视角探针捞得出来。用断言把它钉住。
JOINT_VIEW_PROBES = ("conjunction_in_paren_and_not_grounded",)
assert set(JOINT_VIEW_PROBES) <= {n for n, _ in PROBES}, (
    "联合视角探针被从 PROBES 里删掉了——L5 的判读规则明确要求常驻至少一条"
    "同时读答案与原文的探针，它是单视角探针漏掉的那部分伪影的唯一抓手"
)


@dataclass(frozen=True)
class ProbeScore:
    name: str
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def tpr(self) -> float:
        pos = self.tp + self.fn
        return self.tp / pos if pos else 0.0

    @property
    def fpr(self) -> float:
        neg = self.fp + self.tn
        return self.fp / neg if neg else 0.0

    @property
    def youden_j(self) -> float:
        """|TPR − FPR|。**取绝对值**：把两类判反的特征同样是捷径（模型学到
        取反即可），不取绝对值会让"负例专属特征"显示成 0，看起来干净。"""
        return abs(self.tpr - self.fpr)


def score_pool(pool: list[dict]) -> dict[str, ProbeScore]:
    """**空池是错误，不是"很干净"。**（DeepSeek 审计 #1 与 /code-review 独立
    撞上的同一个洞）空池让七个探针全部 J=0、`gate_failures` 返回空列表——
    "根本没有验收数据"和"验收数据没有捷径"给出完全一样的信号，而这个信号
    守的是花钱那一步。"""
    if not pool:
        raise ValueError(
            "候选池是空的——探针在空池上会全部报 J=0，看起来像「没有捷径」。"
            "先确认池文件是不是没重建/被截断。"
        )
    _assert_kind_matches_shape(pool)
    out: dict[str, ProbeScore] = {}
    for name, fn in PROBES:
        tp = fp = fneg = tn = 0
        for item in pool:
            fired = bool(fn(item))
            if item["label"]:
                tp += fired
                fneg += not fired
            else:
                fp += fired
                tn += not fired
        out[name] = ProbeScore(name, tp, fp, fneg, tn)
    return out


def gate_failures(scores: dict[str, ProbeScore]) -> list[str]:
    """超过硬门槛的**纯结构**探针。内容型探针永远不进这个列表——理由见模块
    顶部注释，不是漏了。"""
    return [
        name for name in STRUCTURE_PROBES
        if name in scores and scores[name].youden_j > STRUCTURE_GATE_J
    ]


def format_table(scores: dict[str, ProbeScore], title: str) -> str:
    n_pos = next(iter(scores.values())).tp + next(iter(scores.values())).fn if scores else 0
    n_neg = next(iter(scores.values())).fp + next(iter(scores.values())).tn if scores else 0
    lines = [
        f"【捷径探针】{title}（正例 {n_pos} / 负例 {n_neg}）",
        f"{'探针':<42}{'TPR':>8}{'FPR':>8}{'J':>8}  门槛",
    ]
    for name, _ in PROBES:
        s = scores[name]
        gated = name in STRUCTURE_PROBES
        mark = f"≤{STRUCTURE_GATE_J}" if gated else "仅报告"
        flag = "  ✗" if gated and s.youden_j > STRUCTURE_GATE_J else ""
        lines.append(f"{name:<42}{s.tpr:>8.1%}{s.fpr:>8.1%}{s.youden_j:>8.3f}  {mark}{flag}")
    return "\n".join(lines)


def report(pool: list[dict], title: str, *, gate: bool = False,
           min_positives: int = 0) -> bool:
    """打印一张表；`gate=True` 时按纯结构探针的硬门槛返回是否通过。

    构建 CLI（`candidate_pool.py`/`adversarial_subset.py`/`train_lora.py
    --make-candidate-data`）都用 `gate=False` 调它——它们的退出码由各自的
    既有判据决定，探针表只是随手打出来看。真正的强制点是本模块的 CLI
    `--gate` 与部署预检，那里才该拦。"""
    scores = score_pool(pool)          # 空池在这里就抛，不会走到"全绿"
    print()
    print(format_table(scores, title))
    if not gate:
        return True
    # **规模也要卡**（/code-review #5）：一份只剩 3 条的截断池同样能让所有探针
    # J=0 而"通过"。文件存在 ≠ 文件够用。
    n_pos = sum(1 for it in pool if it["label"])
    if n_pos < min_positives:
        print(f"  ✗ 正例只有 {n_pos} 条（要求 ≥{min_positives}）——池被截断或没重建完，"
              f"这种规模下探针全绿没有意义")
        return False
    fails = gate_failures(scores)
    if fails:
        print(f"  ✗ 纯结构探针超门槛：{', '.join(fails)} —— 这份池能被不读原文的"
              f"退化分类器分开，不是可以拿去验收的集合")
        return False
    # 措辞是 L5 判读规则的一部分：不许写成"干净/通过"。探针失败不能证明
    # 数据集没有伪影（Feng et al. arXiv:1905.05778），只能证明"没被这几条
    # 探针证伪"。
    print("  ○ 纯结构探针未超线 —— 只是**没被证伪**，不构成「这份池干净」的证据")
    return True


def load_pool(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=Path, default=None, help="候选池 JSON/JSONL")
    ap.add_argument("--gate", action="store_true",
                    help="按纯结构探针的硬门槛决定退出码（冻结验收集才该开）")
    ap.add_argument("--min-positives", type=int, default=0,
                    help="池里至少要有多少条正例，少于此判 fail（防截断池全绿）")
    ap.add_argument("--all", action="store_true",
                    help="三个池的完整基线表：候选级训练池 + 真实 holdout 可信池 + 对抗子集")
    args = ap.parse_args()

    if args.pool is not None:
        # 空池抛的是 ValueError（那是给调用方看的硬错误），但 CLI 被部署预检
        # 调用时应该给一行人话 + 非零退出码，而不是一屏 traceback。
        try:
            ok = report(load_pool(args.pool), str(args.pool),
                        gate=args.gate, min_positives=args.min_positives)
        except ValueError as e:
            print(f"✗ {args.pool}: {e}", file=sys.stderr)
            return 1
        return 0 if ok else 1

    if not args.all:
        ap.print_help()
        return 1

    # 惰性导入：这几个模块要读词表/训练切分，且 `--all` 是本机专用路径。
    from verifier.candidate_pool import load_split
    from verifier.train_lora import build_candidate_training_pool

    ok = True
    train_records = load_split(ROOT / "verifier" / "train.jsonl")
    report(build_candidate_training_pool(train_records), "候选级训练池（只报告，不设门）")
    for name, gate in (("trusted_candidate_pool_holdout.json", False),
                       ("adversarial_subset_holdout.json", True)):
        path = WORK / name
        if not path.exists():
            # **缺文件是失败，不是跳过**（DeepSeek 审计 #1）：原来 `continue`
            # 不动 `ok`，于是"池根本不存在"也能让 --all 退出 0。
            print(f"\n✗ 缺 {path}——先跑对应的重建 CLI。缺池不等于池干净。")
            ok = False
            continue
        ok = report(load_pool(path), name, gate=gate) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
