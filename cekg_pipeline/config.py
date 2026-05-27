import os
from dotenv import load_dotenv

load_dotenv()

# Local vLLM servers (OpenAI-compatible API)
# Main: gpt-oss-120b (120B MoE, MXFP4) on GPU 0 — best single-GPU model
# Mini: qwen3-14b on GPU 1 — used for cheap classification/scene tasks
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:8006/v1")
LLM_MINI_BASE_URL = os.environ.get("LLM_MINI_BASE_URL", "http://localhost:8004/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-oss-120b")
LLM_MINI_MODEL = os.environ.get("LLM_MINI_MODEL", "qwen3-14b")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "EMPTY")

# Legacy names kept for backward compatibility
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "EMPTY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", LLM_MODEL)
BATCH_SIZE = 5
CAUSAL_BATCH_SIZE = 10
SAMPLE_RATE = 0.5
CACHE_MAX_SIZE = 10000

CONTROLLED_ACTION_ONTOLOGY = {
    "call": "name", "label": "name",
    "see": "perceive", "find": "perceive",
    "think": "imagine", "fancy": "imagine",
    "say": "say", "tell": "say", "announce": "say",
    "ask": "demand", "inquire": "demand",
    "warn": "threaten", "intimidate": "threaten",
    "bring": "give", "offer": "give",
    "go": "move", "leave": "move",
    "eat": "eat", "devour": "eat",
    "vow": "promise", "swear": "promise",
    "strike": "attack", "harm": "attack",
    "tremble": "fear", "cry": "fear",
    "look": "watch", "gaze": "watch",
    "symbolize": "represent", "signify": "represent",
}