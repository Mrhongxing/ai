#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate_elder_support_100k.py

生成高质量、风险感知的老年心理支持QA数据集（中文），最多10万条，使用本地Qwen-7B模型。

两种后端：
1) vLLM OpenAI兼容服务器（推荐，速度快）：
    - 启动服务器：
      python -m vllm.entrypoints.openai.api_server --model /path/to/qwen-7b --port 8000
    - 使用 --backend vllm 运行脚本

2) transformers本地加载（较慢，无需服务器）：
    - 使用 --backend hf --hf_model /path/to/qwen-7b 运行脚本

输出（JSONL）：
每行：
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

用法：
  python generate_elder_support_100k.py \
     --backend vllm --base_url http://127.0.0.1:8000/v1 --model_name Qwen \
     --out_dir ./elder_100k --target 100000 --shard_size 5000 --seed 42

注意：
- 这是生成管道，不是医疗系统。
- 安全：遇到自伤风险，回复需支持并鼓励现实寻求帮助。
"""

import argparse  # 导入命令行参数解析库
import datetime as dt  # 导入日期时间库
import hashlib  # 导入哈希库
import json  # 导入JSON处理库
import math  # 导入数学库
import os  # 导入操作系统库
import random  # 导入随机数库
import re  # 导入正则表达式库
import sys  # 导入系统库
import time  # 导入时间库
from collections import Counter, defaultdict, deque  # 导入计数器、默认字典、队列
from dataclasses import dataclass  # 导入数据类装饰器
from typing import Dict, List, Optional, Tuple  # 导入类型提示

# -------------------------
# 风险/类别启发式规则
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
     t = text.strip()  # 去除首尾空白
     for level in (3, 2, 1):  # 从高到低风险
          for pat in RISK_PATTERNS[level]:  # 遍历每个风险模式
                if re.search(pat, t):  # 匹配正则表达式
                     return level  # 返回风险等级
     return 0  # 无风险

def infer_category(text: str, risk_level: int) -> str:
     if risk_level >= 3:  # 高风险直接归类
          return "高风险自伤暗示"
     scores = {}  # 存储类别分数
     for cat, pats in CATEGORY_HINTS.items():  # 遍历类别提示
          s = 0
          for p in pats:
                if re.search(p, text):  # 匹配类别提示
                     s += 1
          scores[cat] = s
     best_cat = max(scores.items(), key=lambda x: x[1])[0]  # 得到最高分类别
     if scores[best_cat] == 0:
          return "日常孤独倾诉"  # 无匹配则归为孤独倾诉
     if risk_level == 2 and best_cat == "日常孤独倾诉":
          return "轻度抑郁低落"  # 风险2且类别为孤独则归为低落
     return best_cat

# -------------------------
# 提示词设计（核心）
# -------------------------
SYSTEM_ASSISTANT = (
     "你是一位温和、有耐心、擅长与老年人沟通的心理支持助手。"
     "目标：提供陪伴式支持、情绪安抚与沟通建议。"
     "要求：语言简单、节奏舒缓、尊重对方；不要说教，不要否定感受。"
     "不要提供医学诊断或处方；必要时建议联系家人或专业机构。"
     "若出现自伤/轻生暗示：优先关心与安全建议，鼓励寻求现实支持与紧急帮助。"
)

USER_GEN_SYSTEM = (
     "你是数据生成器。你要生成“老年人可能说的话”的一句中文求助/倾诉。"
     "必须口语化、真实，不要像写作文。"
     "不要出现任何个人隐私（真实姓名、电话、地址）。"
     "输出只包含一句话，不要解释。"
)

ASSISTANT_GEN_SYSTEM = SYSTEM_ASSISTANT  # 助手生成系统提示

JUDGE_SYSTEM = (
     "你是质量评审员。请根据以下标准给“助手回答”打分0到10（可带一位小数）："
     "1) 是否先共情/确认感受；2) 是否语言适老、简单、温和；3) 是否给出小而可执行的建议；"
     "4) 是否避免说教/指责；5) 若有高风险暗示是否给出安全建议。"
     "只输出一个数字（例如 7.5），不要多余文字。"
)

# -------------------------
# 多样性控制
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

# 目标类别分布（大致）：高风险保持8-12%，不宜过多
TARGET_CATEGORY_RATIO = {
     "日常孤独倾诉": 0.30,
     "家庭关系矛盾": 0.18,
     "衰老与健康焦虑": 0.18,
     "轻度抑郁低落": 0.16,
     "回忆与意义重建": 0.10,
     "高风险自伤暗示": 0.08,
}

def choose_by_ratio(rng: random.Random, ratio: Dict[str, float]) -> str:
     items = list(ratio.items())  # 转为列表
     total = sum(v for _, v in items)  # 总和
     r = rng.random() * total  # 随机数
     s = 0.0
     for k, v in items:
          s += v
          if r <= s:
                return k  # 按比例选择类别
     return items[-1][0]

# 去重辅助：精确哈希+simhash

def sha1(s: str) -> str:
     return hashlib.sha1(s.encode("utf-8")).hexdigest()  # 计算sha1哈希

def tokenize_for_simhash(text: str) -> List[str]:
     t = re.sub(r"\s+", "", text)  # 去除空白
     words = re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]", t)  # 提取英文单词和中文字符
     bigrams = []
     for i in range(len(words) - 1):
          bigrams.append(words[i] + words[i+1])  # 中文字符二元组
     return words + bigrams

def simhash64(tokens: List[str]) -> int:
     v = [0] * 64  # 初始化向量
     for tok in tokens:
          h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)  # md5哈希
          for i in range(64):
                bit = (h >> i) & 1
                v[i] += 1 if bit else -1
     out = 0
     for i in range(64):
          if v[i] > 0:
                out |= (1 << i)
     return out

def hamming64(a: int, b: int) -> int:
     return (a ^ b).bit_count()  # 计算汉明距离

class SimhashDeduper:
     """
     分桶simhash去重器。
     存储simhash签名，拒绝汉明距离小于阈值的近重复。
     """
     def __init__(self, threshold: int = 4):
          self.threshold = threshold  # 汉明距离阈值
          self.buckets: Dict[int, List[int]] = defaultdict(list)  # 分桶存储

     def _bucket_keys(self, sh: int) -> List[int]:
          keys = []
          for k in range(4):
                keys.append((k << 16) | ((sh >> (k*16)) & 0xFFFF))  # 分成4段16位
          return keys

     def is_dup(self, sh: int) -> bool:
          for key in self._bucket_keys(sh):
                for cand in self.buckets.get(key, []):
                     if hamming64(sh, cand) <= self.threshold:
                          return True  # 存在近重复
          return False

     def add(self, sh: int) -> None:
          for key in self._bucket_keys(sh):
                self.buckets[key].append(sh)  # 添加simhash到桶

# 后端：vLLM OpenAI API / HF

class LLMBackend:
     def chat(self, messages: List[Dict[str, str]], max_tokens: int, temperature: float, top_p: float) -> str:
          raise NotImplementedError  # 抽象方法

class VLLMOpenAIBackend(LLMBackend):
     def __init__(self, base_url: str, model_name: str):
          try:
                from openai import OpenAI  # 导入openai客户端
          except Exception as e:
                raise RuntimeError("Please pip install openai") from e
          self.client = OpenAI(base_url=base_url, api_key="EMPTY")  # 初始化客户端
          self.model_name = model_name

     def chat(self, messages: List[Dict[str, str]], max_tokens: int, temperature: float, top_p: float) -> str:
          resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
          )
          return (resp.choices[0].message.content or "").strip()  # 返回生成内容

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
          tok = self.tokenizer
          if hasattr(tok, "apply_chat_template"):
                prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
          else:
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
          if text.startswith(prompt):
                gen = text[len(prompt):].strip()
          else:
                gen = text.strip()
          return gen

# 质量过滤

BAD_PHRASES = [
     "我建议你去死",  # 禁止语句
     "你活该", "别矫情", "想开点", "这有什么", "不就是",  # 冷漠/否定语句
]

def looks_like_essay(user_text: str) -> bool:
     return (len(user_text) > 80 and any(k in user_text for k in ["首先", "其次", "因此", "综上", "然而"]))  # 判断是否像作文

def contains_bad_phrase(text: str) -> bool:
     return any(p in text for p in BAD_PHRASES)  # 检查是否含有不良语句

def empathy_structure_score(answer: str) -> float:
     """
     启发式评分0..1：
     + 共情/确认感受
     + 温和节奏
     + 可执行建议
     """
     s = 0.0
     markers = ["我能感觉到", "听到您这么说", "谢谢您", "我在这里", "很不容易", "您愿意说出来"]
     if any(m in answer for m in markers):
          s += 0.35
     if any(m in answer for m in ["我们先", "不急", "慢慢", "一步", "先把今天"]):
          s += 0.25
     if any(m in answer for m in ["可以试着", "不妨", "今天先", "先做一件小事", "如果方便"]):
          s += 0.25
     if any(m in answer for m in ["你应该", "你必须", "你需要立刻", "听我的"]):
          s -= 0.25
     return max(0.0, min(1.0, s))

def safety_present_if_needed(risk_level: int, answer: str) -> bool:
     if risk_level < 3:
          return True
     must = ["安全", "不独自", "联系", "家人", "专业", "紧急"]
     return sum(1 for m in must if m in answer) >= 2  # 高风险需包含安全词汇

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

# 生成提示

def build_user_gen_prompt(target_category: str, style: str, topic: str, rng: random.Random) -> List[Dict[str, str]]:
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

# IO：分片写入+断点续写

def ensure_dir(p: str) -> None:
     os.makedirs(p, exist_ok=True)  # 创建目录

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

# 主生成循环
@dataclass
class Sample:
     messages: List[Dict[str, str]]  # 消息列表
     risk_level: int  # 风险等级
     category: str  # 类别
     meta: Dict  # 元数据

def make_sample(user_text: str, answer: str, risk: int, cat: str, meta: Dict) -> Sample:
     msgs = [
          {"role": "system", "content": SYSTEM_ASSISTANT},
          {"role": "user", "content": user_text},
          {"role": "assistant", "content": answer},
     ]
     return Sample(messages=msgs, risk_level=risk, category=cat, meta=meta)

def sample_key(user_text: str, answer: str) -> str:
     return sha1(user_text.strip() + "\n---\n" + answer.strip())  # 生成样本唯一键

def main():
     ap = argparse.ArgumentParser()  # 创建参数解析器
     ap.add_argument("--backend", choices=["vllm", "hf"], required=True)  # 后端选择
     ap.add_argument("--base_url", type=str, default="http://127.0.0.1:8000/v1")  # vllm服务器地址
     ap.add_argument("--model_name", type=str, default="Qwen")  # 模型名称
     ap.add_argument("--hf_model", type=str, default="")  # hf模型路径
     ap.add_argument("--out_dir", type=str, default="./elder_100k")  # 输出目录
     ap.add_argument("--target", type=int, default=100000)  # 目标样本数
     ap.add_argument("--shard_size", type=int, default=5000)  # 每片样本数
     ap.add_argument("--seed", type=int, default=42)  # 随机种子

     ap.add_argument("--user_temp", type=float, default=0.95)  # 用户生成温度
     ap.add_argument("--user_top_p", type=float, default=0.95)  # 用户生成top_p
     ap.add_argument("--ans_temp", type=float, default=0.7)  # 答案生成温度
     ap.add_argument("--ans_top_p", type=float, default=0.9)  # 答案生成top_p
     ap.add_argument("--judge", action="store_true", help="启用模型评审（更慢但质量更好）。")
     ap.add_argument("--judge_min", type=float, default=7.0)  # 评审最低分

     ap.add_argument("--simhash_thresh", type=int, default=4)  # simhash阈值
     ap.add_argument("--max_retries", type=int, default=8)  # 最大重试次数

     args = ap.parse_args()  # 解析参数

     rng = random.Random(args.seed)  # 初始化随机数生成器
     ensure_dir(args.out_dir)  # 确保输出目录存在

     if args.backend == "vllm":
          backend: LLMBackend = VLLMOpenAIBackend(args.base_url, args.model_name)  # 初始化vllm后端
     else:
          if not args.hf_model:
                raise SystemExit("--hf_model is required when --backend hf")
          backend = HFBackend(args.hf_model)  # 初始化hf后端

     existing = count_existing_lines(args.out_dir)  # 已有样本数
     target_total = args.target  # 目标总数
     if existing >= target_total:
          print(f"Already have {existing} samples >= target {target_total}. Nothing to do.")
          return

     exact_seen = set()  # 存储精确哈希
     deduper = SimhashDeduper(threshold=args.simhash_thresh)  # 初始化simhash去重器

     shard_idx = next_shard_index(args.out_dir)  # 下一个分片索引
     cur_path = os.path.join(args.out_dir, f"shard_{shard_idx:04d}.jsonl")  # 当前分片路径
     cur_f = open(cur_path, "a", encoding="utf-8")  # 打开分片文件
     cur_count = 0  # 当前分片计数

     kept_counter = Counter()  # 记录类别分布
     tried = 0  # 尝试次数
     kept = 0  # 保留样本数
     start = time.time()  # 开始时间

     def rotate_shard():
          nonlocal shard_idx, cur_path, cur_f, cur_count
          cur_f.flush()
          cur_f.close()
          shard_idx += 1
          cur_path = os.path.join(args.out_dir, f"shard_{shard_idx:04d}.jsonl")
          cur_f = open(cur_path, "a", encoding="utf-8")
          cur_count = 0

     def should_accept_category(cat: str) -> bool:
          total = sum(kept_counter.values()) + 1e-9
          cur_ratio = kept_counter[cat] / total
          target_ratio = TARGET_CATEGORY_RATIO.get(cat, 0.1)
          if cur_ratio < target_ratio:
                return True
          over = cur_ratio / max(target_ratio, 1e-6)
          p = 1.0 / min(2.5, over)
          return rng.random() < p

     try:
          while existing + kept < target_total:
                target_cat = choose_by_ratio(rng, TARGET_CATEGORY_RATIO)  # 按比例选择类别
                style = rng.choice(USER_STYLES)  # 随机选择风格
                topic = rng.choice(TOPICS)  # 随机选择主题

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
                     user_text = re.split(r"[\n\r]+", user_text)[0].strip()
                     user_text = re.sub(r"(解释|原因|分析)[:：].*$", "", user_text).strip()

                     if 6 <= len(user_text) <= 80 and not looks_like_essay(user_text):
                          break
                     user_text = ""
                if not user_text:
                     tried += 1
                     continue

                risk = infer_risk_level(user_text)  # 推断风险等级
                cat = infer_category(user_text, risk)  # 推断类别

                if not should_accept_category(cat):  # 类别平衡
                     tried += 1
                     continue

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
                     ans = re.sub(r"\n{3,}", "\n\n", ans)
                     if ans:
                          break
                if not ans:
                     tried += 1
                     continue

                if not basic_quality_gate(user_text, ans, risk):  # 基本质量过滤
                     tried += 1
                     continue

                score = None
                if args.judge:
                     score = judge_score(backend, user_text, ans)
                     if score is None or score < args.judge_min:
                          tried += 1
                          continue

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

                if cur_count >= args.shard_size:
                     rotate_shard()

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
     main()  # 程序入口
