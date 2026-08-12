#!/usr/bin/env python3
"""补充：span 级候选统计——两段式（规则出候选→判别器判候选）的样本量测算"""
import json, re, unicodedata
from pathlib import Path

REPO = Path('/Users/chenhao/ClaudeCode/elder-trans-run1')

def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]

def parse_input(inp):
    src = inp.split('【回答】')[0].replace('【原文】', '')
    ans = inp.split('【回答】')[1].split('【要点】')[0]
    return src, ans

jargon = set()
for l in open(REPO/'lexicon/jargon.txt'):
    l = l.strip()
    if l and not l.startswith('#') and len(l) >= 3:
        jargon.add(l)

NUM_RE = re.compile(r'\d+(?:\.\d+)?')

def norm(s):
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', s))

tot_lex = tot_num = 0
cases_with_cand = 0
span_pos = 0   # 候选词命中 gold evidence（大致=正例 span）
span_pos_cases = 0
fail_flagged = 0
n = 0
for split in ['train.jsonl', 'holdout.jsonl']:
    for r in load_jsonl(REPO/'verifier'/split):
        n += 1
        src, ans = parse_input(r['input'])
        gold = json.loads(r['output'])
        nsrc, nans = norm(src), norm(ans)
        hits_lex = {w for w in jargon if w in nans and w not in nsrc}
        hits_num = {m for m in NUM_RE.findall(nans) if m not in nsrc}
        tot_lex += len(hits_lex); tot_num += len(hits_num)
        if hits_lex or hits_num:
            cases_with_cand += 1
        evs = [norm(rl['evidence']) for rl in gold['red_lines'] if rl['violated']]
        if evs:
            hit_in_ev = {w for w in (hits_lex|hits_num) if any(w in e for e in evs)}
            if hits_lex or hits_num:
                fail_flagged += 1
            if hit_in_ev:
                span_pos += len(hit_in_ev)
                span_pos_cases += 1

print(f'全池 {n} 条（train+holdout）')
print(f'候选 span 总数: 词表 {tot_lex} + 数字 {tot_num} = {tot_lex+tot_num}')
print(f'  平均每条 case {(tot_lex+tot_num)/n:.1f} 个候选；有候选的 case: {cases_with_cand}')
print(f'fail 案例中被规则打上旗标的: {fail_flagged}')
print(f'候选词直接命中 gold evidence 的 span 数: {span_pos}（分布在 {span_pos_cases} 条 case）')
print()
print('→ 两段式测算：span 级训练点 ≈ 候选总数（负例=未命中 evidence 的候选，正例≈命中 evidence 的候选）')
