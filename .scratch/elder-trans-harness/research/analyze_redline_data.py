#!/usr/bin/env python3
"""盘内数据速查（ticket 12 研究阶段，只读）：
1) 违规实例按红线 idx 的分布（train/holdout 分开）
2) 最优配置（f5 / 4b_f5）在 150 条网格子集上的 FN 剖面
3) 规则基线上界估计：词表(药名/实体) + 数字 两条规则在 holdout 499 条上的
   case 级召回/误报
"""
import json, re, sys, unicodedata
from collections import Counter
from pathlib import Path

REPO = Path('/Users/chenhao/ClaudeCode/elder-trans-run1')

def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]

def parse_input(inp):
    src = inp.split('【回答】')[0].replace('【原文】', '')
    ans = inp.split('【回答】')[1].split('【要点】')[0]
    return src, ans

# ---------- 1) 红线分布 ----------
def redline_dist(rows, name):
    n_fail = 0
    viol = Counter()
    n_rl = Counter()
    for r in rows:
        gold = json.loads(r['output'])
        n_rl[len(gold['red_lines'])] += 1
        if gold['verdict'] == 'fail':
            n_fail += 1
        for rl in gold['red_lines']:
            if rl['violated']:
                viol[rl['idx']] += 1
    total_viol = sum(viol.values())
    print(f'[{name}] rows={len(rows)}  fail_cases={n_fail}  violation_instances={total_viol}')
    print(f'  每条 case 的红线条数分布: {dict(n_rl)}')
    for idx in sorted(viol):
        print(f'  红线{idx}: {viol[idx]}  ({viol[idx]/total_viol:.1%})')
    return viol

train = load_jsonl(REPO/'verifier/train.jsonl')
holdout = load_jsonl(REPO/'verifier/holdout.jsonl')
print('==== 1) 违规实例按红线分布 ====')
v_tr = redline_dist(train, 'train 1456')
v_ho = redline_dist(holdout, 'holdout 499')

# ---------- 2) FN 剖面 ----------
print()
print('==== 2) FN 剖面（150 条网格子集，gold fail & pred pass）====')
gridsub = {r['case_id']: r for r in load_jsonl(REPO/'verifier/work/holdout_gridsub.jsonl')}
for predfile in ['pred_f5_grid.jsonl', 'pred_4b_f5_grid.jsonl']:
    preds = {r['case_id']: r for r in load_jsonl(REPO/'verifier/work'/predfile)}
    fn_by_rl = Counter(); tp_by_rl = Counter()
    fn_cases, tp_cases = [], []
    for cid, g in gridsub.items():
        gold = json.loads(g['output'])
        if gold['verdict'] != 'fail':
            continue
        p = preds.get(cid)
        pred_fail = p is not None and p.get('verdict') == 'fail'
        gold_idx = [rl['idx'] for rl in gold['red_lines'] if rl['violated']]
        if pred_fail:
            tp_cases.append(cid)
            for i in gold_idx: tp_by_rl[i] += 1
        else:
            fn_cases.append(cid)
            for i in gold_idx: fn_by_rl[i] += 1
    print(f'[{predfile}] TP={len(tp_cases)} FN={len(fn_cases)}')
    for idx in sorted(set(fn_by_rl)|set(tp_by_rl)):
        t, f = tp_by_rl[idx], fn_by_rl[idx]
        print(f'  红线{idx}: 抓到 {t} / 漏掉 {f}  (该红线漏报率 {f/(t+f):.1%})')
    # FN 案例的 gold evidence 模式速览
    if predfile == 'pred_f5_grid.jsonl':
        print('  FN 案例 gold evidence 抽样（前 12 条）:')
        for cid in fn_cases[:12]:
            gold = json.loads(gridsub[cid]['output'])
            evs = [(rl['idx'], rl['evidence'][:40]) for rl in gold['red_lines'] if rl['violated']]
            print(f'    {cid}: {evs}')

# ---------- 3) 规则基线 ----------
print()
print('==== 3) 规则基线上界（holdout 全量 499 条）====')

# 词表：跳过注释行；只留 >=3 字条目（2 字条目误报太多）
jargon = set()
for l in open(REPO/'lexicon/jargon.txt'):
    l = l.strip()
    if l and not l.startswith('#') and len(l) >= 3:
        jargon.add(l)
print(f'词表条目(≥3字): {len(jargon)}')

NUM_RE = re.compile(r'\d+(?:\.\d+)?')

def norm(s):
    # 全角→半角，去空白，便于 substring 比对
    s = unicodedata.normalize('NFKC', s)
    return re.sub(r'\s+', '', s)

def rule_flags(src, ans):
    nsrc, nans = norm(src), norm(ans)
    # R1 药名/实体越界：词表词出现在 answer 但不在 source
    hits_lex = sorted({w for w in jargon if w in nans and w not in nsrc})
    # R2 数字越界：answer 里的数字串不在 source
    hits_num = sorted({m for m in NUM_RE.findall(nans) if m not in nsrc})
    return hits_lex, hits_num

# 词表逐条 in 扫描 499 条太慢(4.3万词×499)？先试速度，必要时用 Aho-Corasick 或分块
import time
t0 = time.time()
res = []
for r in holdout:
    src, ans = parse_input(r['input'])
    gold = json.loads(r['output'])
    hits_lex, hits_num = rule_flags(src, ans)
    res.append((r['case_id'], gold, hits_lex, hits_num))
print(f'扫描耗时 {time.time()-t0:.1f}s')

def score(res, use_lex, use_num, name):
    tp=fn=fp=tn=0
    fn_ids=[]
    for cid, gold, hl, hn in res:
        flag = (use_lex and bool(hl)) or (use_num and bool(hn))
        is_fail = gold['verdict']=='fail'
        if is_fail and flag: tp+=1
        elif is_fail: fn+=1; fn_ids.append(cid)
        elif flag: fp+=1
        else: tn+=1
    print(f'[{name}] TP={tp} FN={fn} FP={fp} TN={tn}  漏报率={fn/(tp+fn):.1%}  误报率={fp/(fp+tn):.1%}')
    return fn_ids

score(res, True, False, 'R1 词表越界')
score(res, False, True, 'R2 数字越界')
fn_ids = score(res, True, True, 'R1∪R2')

# 漏掉的 fail 案例长什么样
print('R1∪R2 漏掉的 fail 案例 gold evidence（全部）:')
by_id = {c: (g,hl,hn) for c,g,hl,hn in res}
for cid in fn_ids:
    g,_,_ = by_id[cid]
    evs = [(rl['idx'], rl['evidence'][:50]) for rl in g['red_lines'] if rl['violated']]
    print(f'  {cid}: {evs}')
