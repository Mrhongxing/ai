#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate_elder_support_100k.py

Generate a high-quality, risk-aware elderly mental health supportive QA dataset
(Chinese) up to 100,000 samples using a LOCAL Qwen-7B model.

Two backends:
1) vLLM OpenAI-compatible server (recommended, fast):
   - start server:
     python -m vllm.entrypoints.openai.api_server --model /path/to/qwen-7b --port 8000
   - run script with --backend vllm

2) transformers local load (slower, but no server):
   - run script with --backend hf --hf_model /path/to/qwen-7b

Outputs (JSONL):
Each line:
{
  "messages": [...],
  "risk_level": 0-3,
  "category": "...",
  "meta": {
     "user_style": "...",
     "topic": "...",
     "seed": "...",
     "score": float,
     "backend": "...",
     "ts": "..."
  }
}

Usage:
  python generate_elder_support_100k.py \
    --backend vllm --base_url http://127.0.0.1:8000/v1 --model_name Qwen \
    --out_dir ./elder_100k --target 100000 --shard_size 5000 --seed 42

Notes:
- This is a generation pipeline, not a medical system.
- Safety: For self-harm risk, respond with support + encourage real-world help.
"""

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# -------------------------
# Risk/category heuristics
# -------------------------
RISK_PATTERNS = {
    3: [
        r"不想活了", r"想死", r"轻生", r"结束生命", r"自杀", r"我准备.*(死|走)",
        r"走了算了", r"一了百了", r"不如死了", r"活着没意思.*(想|要)走"
    ],
    2: [
        r"活着没意思", r"没意义", r"绝望", r"撑不下去", r"心里很空",
        r"每天都难受", r"一点希望都没有", r"不想见人", r"什么都不想做"
    ],
    1: [
        r"难过", r"委屈", r"心烦", r"焦虑", r"睡不着", r"害怕",
        r"担心", r"孤独", r"想哭", r"烦躁"
    ],
}

CATEGORIES = [
    "日常孤独倾诉",
    "家庭关系矛盾",
    "衰老与健康焦虑",
    "轻度抑郁低落",
    "回忆与意义重建",
    "高风险自伤暗示",
]

CATEGORY_HINTS = {
    "日常孤独倾诉": [r"孤独", r"没人陪", r"一个人", r"冷清", r"寂寞", r"说说话"],
    "家庭关系矛盾": [r"孩子", r"儿子", r"女儿", r"老伴", r"家里", r"吵", r"不理解", r"不来看我", r"孙子"],
    "衰老与健康焦虑": [r"身体", r"生病", r"疼", r"医院", r"老了", r"记性", r"走不动", r"慢性病", r"血压", r"糖尿病"],
    "轻度抑郁低落": [r"没劲", r"提不起", r"不想", r"空", r"没意义", r"难熬", r"累", r"麻木"],
    "回忆与意义重建": [r"以前", r"回忆", r"年轻时", r"过去", r"那时候", r"老朋友", r"一辈子", r"当年"],
    "高风险自伤暗示": [r"不想活", r"想死", r"自杀", r"结束生命", r"轻生", r"一了百了"],
}

def infer_risk_level(text: str) -> int:
    t = text.strip()
    for level in (3, 2, 1):
        for pat in RISK_PATTERNS[level]:
            if re.search(pat, t):
                return level
    return 0

def infer_category(text: str, risk_level: int) -> str:
    if risk_level >= 3:
        return "高风险自伤暗示"
    scores = {}
    for cat, pats in CATEGORY_HINTS.items():
        s = 0
        for p in pats:
            if re.search(p, text):
                s += 1
        scores[cat] = s
    best_cat = max(scores.items(), key=lambda x: x[1])[0]
    if scores[best_cat] == 0:
        return "日常孤独倾诉"
    # If risk_level==2 and best_cat isn't high-risk, keep as low mood category
    if risk_level == 2 and best_cat == "日常孤独倾诉":
        return "轻度抑郁低落"
    return best_cat

# -------------------------
# Prompt design (core!)
# -------------------------
SYSTEM_ASSISTANT = (
    "你是一位温和、有耐心、擅长与老年人沟通的心理支持助手。"
    "目标：提供陪伴式支持、情绪安抚与沟通建议。"
    "要求：语言简单、节奏舒缓、尊重对方；不要说教，不要否定感受。"
    "不要提供医学诊断或处方；必要时建议联系家人或专业机构。"
    "若出现自伤/轻生暗示：优先关心与安全建议，鼓励寻求现实支持与紧急帮助。"
)

# User query generator: force diversity + realism
USER_GEN_SYSTEM = (
    "你是数据生成器。你要生成“老年人可能说的话”的一句中文求助/倾诉。"
    "必须口语化、真实，不要像写作文。"
    "不要出现任何个人隐私（真实姓名、电话、地址）。"
    "输出只包含一句话，不要解释。"
)

# This rubric creates a stable supportive style.
ASSISTANT_GEN_SYSTEM = SYSTEM_ASSISTANT

# Quality judge (optional): the same model scores outputs; not perfect but helps filtering
JUDGE_SYSTEM = (
    "你是质量评审员。请根据以下标准给“助手回答”打分0到10（可带一位小数）："
    "1) 是否先共情/确认感受；2) 是否语言适老、简单、温和；3) 是否给出小而可执行的建议；"
    "4) 是否避免说教/指责；5) 若有高风险暗示是否给出安全建议。"
    "只输出一个数字（例如 7.5），不要多余文字。"
)

# -------------------------
# Diversity controls
# -------------------------
USER_STYLES = [
    "朴素直白", "含蓄克制", "唠叨碎碎念", "有点生气", "有点害怕", "自责内疚",
    "半开玩笑", "反复强调", "语气很轻", "偏方言口语(轻度)"
]

TOPICS = [
    "独居与孤独", "丧偶与思念", "子女忙不联系", "与子女冲突", "照顾孙辈压力",
    "慢性病焦虑", "睡眠问题", "记忆变差担心", "行动不便害怕摔倒",
    "退休失落", "朋友变少", "回忆往事", "自我价值感低", "节日更孤单",
    "高风险绝望表达"
]

# Target distribution (rough): keep high-risk ~8-12% but not overwhelming
TARGET_CATEGORY_RATIO = {
    "日常孤独倾诉": 0.30,
    "家庭关系矛盾": 0.18,
    "衰老与健康焦虑": 0.18,
    "轻度抑郁低落": 0.16,
    "回忆与意义重建": 0.10,
    "高风险自伤暗示": 0.08,
}

def choose_by_ratio(rng: random.Random, ratio: Dict[str, float]) -> str:
    items = list(ratio.items())
    total = sum(v for _, v in items)
    r = rng.random() * total
    s = 0.0
    for k, v in items:
        s += v
        if r <= s:
            return k
    return items[-1][0]

# Dedup helpers: exact hash + simhash

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def tokenize_for_simhash(text: str) -> List[str]:
    # Simple tokenization: Chinese char bigrams + alnum words
    t = re.sub(r"\s+", "", text)
    words = re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]", t)
    # bigrams for Chinese chars
    bigrams = []
    for i in range(len(words) - 1):
        bigrams.append(words[i] + words[i+1])
    return words + bigrams

def simhash64(tokens: List[str]) -> int:
    # Classic simhash
    v = [0] * 64
    for tok in tokens:
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
        for i in range(64):
            bit = (h >> i) & 1
            v[i] += 1 if bit else -1
    out = 0
    for i in range(64):
        if v[i] > 0:
            out |= (1 << i)
    return out

def hamming64(a: int, b: int) -> int:
    return (a ^ b).bit_count()

class SimhashDeduper:
    """
    Bucketed simhash deduper.
    Stores simhash signatures and rejects near-duplicates within small Hamming distance.
    """
    def __init__(self, threshold: int = 4):
        self.threshold = threshold
        self.buckets: Dict[int, List[int]] = defaultdict(list)

    def _bucket_keys(self, sh: int) -> List[int]:
        # Split into 4 chunks of 16 bits
        keys = []
        for k in range(4):
            keys.append((k << 16) | ((sh >> (k*16)) & 0xFFFF))
        return keys

    def is_dup(self, sh: int) -> bool:
        for key in self._bucket_keys(sh):
            for cand in self.buckets.get(key, []):
                if hamming64(sh, cand) <= self.threshold:
                    return True
        return False

    def add(self, sh: int) -> None:
        for key in self._bucket_keys(sh):
            self.buckets[key].append(sh)

# Backend: vLLM OpenAI API / HF

class LLMBackend:
    def chat(self, messages: List[Dict[str, str]], max_tokens: int, temperature: float, top_p: float) -> str:
        raise NotImplementedError

class VLLMOpenAIBackend(LLMBackend):
    def __init__(self, base_url: str, model_name: str):
        # Uses openai python client with custom base_url
        try:
            from openai import OpenAI
        except Exception as e:
            raise RuntimeError("Please pip install openai") from e
        self.client = OpenAI(base_url=base_url, api_key="EMPTY")
        self.model_name = model_name

    def chat(self, messages: List[Dict[str, str]], max_tokens: int, temperature: float, top_p: float) -> str:
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()

class HFBackend(LLMBackend):
    def __init__(self, model_path: str, device: str = "cuda"):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as e:
            raise RuntimeError("Please pip install transformers accelerate torch") from e
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto" if device == "cuda" else None
        )
        self.model.eval()

    def chat(self, messages: List[Dict[str, str]], max_tokens: int, temperature: float, top_p: float) -> str:
        # Many Qwen chat models support apply_chat_template; fallback to naive concat if not available.
        tok = self.tokenizer
        if hasattr(tok, "apply_chat_template"):
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            # naive
            prompt = ""
            for m in messages:
                prompt += f"{m['role'].upper()}: {m['content']}\n"
            prompt += "ASSISTANT:"

        inputs = tok(prompt, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                eos_token_id=tok.eos_token_id,
            )
        text = tok.decode(out[0], skip_special_tokens=True)
        # Extract newly generated portion
        if text.startswith(prompt):
            gen = text[len(prompt):].strip()
        else:
            gen = text.strip()
        return gen

# Quality filters

BAD_PHRASES = [
    "我建议你去死",  # obviously disallowed
    "你活该", "别矫情", "想开点", "这有什么", "不就是",  # harsh / invalidating
]

def looks_like_essay(user_text: str) -> bool:
    # Too formal, long, with many "首先其次因此"
    return (len(user_text) > 80 and any(k in user_text for k in ["首先", "其次", "因此", "综上", "然而"]))

def contains_bad_phrase(text: str) -> bool:
    return any(p in text for p in BAD_PHRASES)

def empathy_structure_score(answer: str) -> float:
    """
    heuristic score 0..1:
    + empathy/validation markers
    + gentle pacing
    + actionable micro-suggestion
    """
    s = 0.0
    markers = ["我能感觉到", "听到您这么说", "谢谢您", "我在这里", "很不容易", "您愿意说出来"]
    if any(m in answer for m in markers):
        s += 0.35
    if any(m in answer for m in ["我们先", "不急", "慢慢", "一步", "先把今天"]):
        s += 0.25
    if any(m in answer for m in ["可以试着", "不妨", "今天先", "先做一件小事", "如果方便"]):
        s += 0.25
    # penalize lecturing
    if any(m in answer for m in ["你应该", "你必须", "你需要立刻", "听我的"]):
        s -= 0.25
    return max(0.0, min(1.0, s))

def safety_present_if_needed(risk_level: int, answer: str) -> bool:
    if risk_level < 3:
        return True
    # require safety language for high risk
    must = ["安全", "不独自", "联系", "家人", "专业", "紧急"]
    return sum(1 for m in must if m in answer) >= 2

def length_ok(user_text: str, answer: str) -> bool:
    if not (6 <= len(user_text) <= 80):
        return False
    if not (80 <= len(answer) <= 500):
        return False
    return True

def basic_quality_gate(user_text: str, answer: str, risk_level: int) -> bool:
    if looks_like_essay(user_text):
        return False
    if contains_bad_phrase(user_text) or contains_bad_phrase(answer):
        return False
    if not length_ok(user_text, answer):
        return False
    if empathy_structure_score(answer) < 0.55:
        return False
    if not safety_present_if_needed(risk_level, answer):
        return False
    return True

# Optional model-based judge
def judge_score(backend: LLMBackend, user_text: str, answer: str) -> Optional[float]:
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": f"【用户】{user_text}\n【助手】{answer}\n请打分："}
    ]
    try:
        out = backend.chat(messages, max_tokens=16, temperature=0.0, top_p=1.0)
        m = re.search(r"(\d+(\.\d+)?)", out)
        if not m:
            return None
        val = float(m.group(1))
        return max(0.0, min(10.0, val))
    except Exception:
        return None

# Generation prompts

def build_user_gen_prompt(target_category: str, style: str, topic: str, rng: random.Random) -> List[Dict[str, str]]:
    # steer to target category via instructions, but keep it natural
    steer = (
        f"生成一句老年人求助/倾诉：\n"
        f"- 目标主题：{topic}\n"
        f"- 语气风格：{style}\n"
        f"- 倾向类别：{target_category}\n"
        f"要求：一句话，口语化，真实，不要写成文章；不出现个人隐私。"
    )
    return [
        {"role": "system", "content": USER_GEN_SYSTEM},
        {"role": "user", "content": steer},
    ]

def build_assistant_gen_prompt(user_text: str, risk_level: int) -> List[Dict[str, str]]:
    # For high-risk, explicitly require safety guidance without details
    extra = ""
    if risk_level >= 3:
        extra = (
            "注意：用户可能有自伤/轻生暗示。回答必须："
            "1) 先共情与关心；2) 提醒安全、尽量不要独自一人；"
            "3) 鼓励联系家人/朋友/专业机构或紧急求助；"
            "4) 不提供任何自伤方法细节。"
        )
    else:
        extra = (
            "回答结构建议：1) 共情确认；2) 轻柔提问(可选)；3) 给一个小而可执行的建议；4) 温和收尾。"
        )
    return [
        {"role": "system", "content": ASSISTANT_GEN_SYSTEM},
        {"role": "user", "content": f"【用户】{user_text}\n{extra}\n请用中文回答："},
    ]

# IO: shard writer + resume

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def count_existing_lines(out_dir: str) -> int:
    if not os.path.isdir(out_dir):
        return 0
    total = 0
    for fn in os.listdir(out_dir):
        if fn.endswith(".jsonl") and fn.startswith("shard_"):
            path = os.path.join(out_dir, fn)
            with open(path, "r", encoding="utf-8") as f:
                for _ in f:
                    total += 1
    return total

def next_shard_index(out_dir: str) -> int:
    if not os.path.isdir(out_dir):
        return 0
    mx = -1
    for fn in os.listdir(out_dir):
        m = re.match(r"shard_(\d+)\.jsonl$", fn)
        if m:
            mx = max(mx, int(m.group(1)))
    return mx + 1

# Main generation loop
@dataclass
class Sample:
    messages: List[Dict[str, str]]
    risk_level: int
    category: str
    meta: Dict

def make_sample(user_text: str, answer: str, risk: int, cat: str, meta: Dict) -> Sample:
    msgs = [
        {"role": "system", "content": SYSTEM_ASSISTANT},
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": answer},
    ]
    return Sample(messages=msgs, risk_level=risk, category=cat, meta=meta)

def sample_key(user_text: str, answer: str) -> str:
    return sha1(user_text.strip() + "\n---\n" + answer.strip())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["vllm", "hf"], required=True)
    ap.add_argument("--base_url", type=str, default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model_name", type=str, default="Qwen")
    ap.add_argument("--hf_model", type=str, default="")
    ap.add_argument("--out_dir", type=str, default="./elder_100k")
    ap.add_argument("--target", type=int, default=100000)
    ap.add_argument("--shard_size", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)

    # generation params
    ap.add_argument("--user_temp", type=float, default=0.95)
    ap.add_argument("--user_top_p", type=float, default=0.95)
    ap.add_argument("--ans_temp", type=float, default=0.7)
    ap.add_argument("--ans_top_p", type=float, default=0.9)
    ap.add_argument("--judge", action="store_true", help="Enable model-based judging (slower but better quality).")
    ap.add_argument("--judge_min", type=float, default=7.0)

    # dedup / quality
    ap.add_argument("--simhash_thresh", type=int, default=4)
    ap.add_argument("--max_retries", type=int, default=8)

    args = ap.parse_args()

    rng = random.Random(args.seed)
    ensure_dir(args.out_dir)

    # backend init
    if args.backend == "vllm":
        backend: LLMBackend = VLLMOpenAIBackend(args.base_url, args.model_name)
    else:
        if not args.hf_model:
            raise SystemExit("--hf_model is required when --backend hf")
        backend = HFBackend(args.hf_model)

    # resume count
    existing = count_existing_lines(args.out_dir)
    target_total = args.target
    if existing >= target_total:
        print(f"Already have {existing} samples >= target {target_total}. Nothing to do.")
        return

    # Prepare dedup
    exact_seen = set()  # store sha1 keys (can grow; acceptable for 100k)
    deduper = SimhashDeduper(threshold=args.simhash_thresh)

    # If resuming, we should load previous keys lightly (optional).
    # For speed, we won't parse all old shards by default. If you need strict cross-run dedup,
    # add a scan step. Here we accept minor duplicates across runs.
    # (You can set --strict_resume in an extended version.)

    shard_idx = next_shard_index(args.out_dir)
    cur_path = os.path.join(args.out_dir, f"shard_{shard_idx:04d}.jsonl")
    cur_f = open(cur_path, "a", encoding="utf-8")
    cur_count = 0

    # tracking distribution for balancing
    kept_counter = Counter()
    tried = 0
    kept = 0
    start = time.time()

    def rotate_shard():
        nonlocal shard_idx, cur_path, cur_f, cur_count
        cur_f.flush()
        cur_f.close()
        shard_idx += 1
        cur_path = os.path.join(args.out_dir, f"shard_{shard_idx:04d}.jsonl")
        cur_f = open(cur_path, "a", encoding="utf-8")
        cur_count = 0

    def should_accept_category(cat: str) -> bool:
        # soft balancing: accept more if under target ratio
        total = sum(kept_counter.values()) + 1e-9
        cur_ratio = kept_counter[cat] / total
        target_ratio = TARGET_CATEGORY_RATIO.get(cat, 0.1)
        # If under target, accept with higher prob; if over, accept with lower prob
        if cur_ratio < target_ratio:
            return True
        # over target -> probabilistic throttling
        over = cur_ratio / max(target_ratio, 1e-6)
        p = 1.0 / min(2.5, over)  # cap throttling
        return rng.random() < p

    try:
        while existing + kept < target_total:
            # choose target category/topic/style to drive diversity
            target_cat = choose_by_ratio(rng, TARGET_CATEGORY_RATIO)
            style = rng.choice(USER_STYLES)
            topic = rng.choice(TOPICS)

            # Generate user text with retries
            user_text = ""
            for _ in range(args.max_retries):
                user_msgs = build_user_gen_prompt(target_cat, style, topic, rng)
                user_text = backend.chat(
                    user_msgs,
                    max_tokens=64,
                    temperature=args.user_temp,
                    top_p=args.user_top_p
                )
                user_text = re.sub(r"^['\"“”\s]+|['\"“”\s]+$", "", user_text).strip()
                # enforce "one sentence"
                user_text = re.split(r"[\n\r]+", user_text)[0].strip()
                # drop trailing explanations
                user_text = re.sub(r"(解释|原因|分析)[:：].*$", "", user_text).strip()

                if 6 <= len(user_text) <= 80 and not looks_like_essay(user_text):
                    break
                user_text = ""
            if not user_text:
                tried += 1
                continue

            risk = infer_risk_level(user_text)
            cat = infer_category(user_text, risk)

            # category throttle to balance
            if not should_accept_category(cat):
                tried += 1
                continue

            # Generate assistant answer
            ans = ""
            for _ in range(args.max_retries):
                ans_msgs = build_assistant_gen_prompt(user_text, risk)
                ans = backend.chat(
                    ans_msgs,
                    max_tokens=420,
                    temperature=args.ans_temp,
                    top_p=args.ans_top_p
                )
                ans = ans.strip()
                # basic cleanup
                ans = re.sub(r"\n{3,}", "\n\n", ans)
                if ans:
                    break
            if not ans:
                tried += 1
                continue

            # basic gate
            if not basic_quality_gate(user_text, ans, risk):
                tried += 1
                continue

            # optional model-based judging
            score = None
            if args.judge:
                score = judge_score(backend, user_text, ans)
                if score is None or score < args.judge_min:
                    tried += 1
                    continue

            # dedup checks
            key = sample_key(user_text, ans)
            if key in exact_seen:
                tried += 1
                continue
            sh = simhash64(tokenize_for_simhash(user_text + " " + ans))
            if deduper.is_dup(sh):
                tried += 1
                continue

            exact_seen.add(key)
            deduper.add(sh)

            meta = {
                "user_style": style,
                "topic": topic,
                "target_category": target_cat,
                "score": score,
                "backend": args.backend,
                "ts": dt.datetime.now().isoformat(timespec="seconds"),
            }
            sample = make_sample(user_text, ans, risk, cat, meta)

            cur_f.write(json.dumps({
                "messages": sample.messages,
                "risk_level": sample.risk_level,
                "category": sample.category,
                "meta": sample.meta
            }, ensure_ascii=False) + "\n")
            cur_count += 1

            kept += 1
            kept_counter[cat] += 1

            # rotate shard
            if cur_count >= args.shard_size:
                rotate_shard()

            # progress log
            tried += 1
            if kept % 200 == 0:
                elapsed = time.time() - start
                rate = kept / max(elapsed, 1e-6)
                done = existing + kept
                print(f"[{done}/{target_total}] kept={kept} tried={tried} rate={rate:.2f}/s "
                      f"dist={dict(kept_counter)}",
                      flush=True)

        print("Generation complete.")
        print("Final dist:", dict(kept_counter))

        # write a report
        report_path = os.path.join(args.out_dir, "report.txt")
        total_new = kept
        total_all = existing + kept
        with open(report_path, "w", encoding="utf-8") as rf:
            rf.write(f"Total existing: {existing}\n")
            rf.write(f"Total newly generated: {total_new}\n")
            rf.write(f"Total overall: {total_all}\n\n")
            rf.write("Category distribution (new):\n")
            for c in CATEGORIES:
                rf.write(f"  {c}: {kept_counter[c]}\n")
        print("Report:", report_path)

    finally:
        try:
            cur_f.flush()
            cur_f.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()