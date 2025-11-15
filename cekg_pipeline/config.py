import os
from dotenv import load_dotenv

# --- Load environment variables ---
# This line looks for a .env file in the project root and loads it.
load_dotenv()
# ----------------------------------

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini") # Keeps default
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