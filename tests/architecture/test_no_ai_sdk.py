"""Production code does not use an LLM or another hosted AI model (design 2026-11 M5, step S12; tech lead
2026-11-15: risk and execution are deterministic code, no LLM in the trading path).

Forbidden in trading, runner, lab, data, infra, monitoring and scripts:
- importing an LLM or AI-service SDK or framework (``import``, ``from ... import``, ``importlib.import_module``
  or ``__import__`` with a literal name): the top-level packages in ``AI_PACKAGES``, any ``langchain*`` or
  ``llama_index*`` package, and ``google.generativeai``, ``google.genai``, ``google.ai.generativelanguage``;
- a string naming an AI service's API host (``AI_HOSTS``), which would call the model over plain HTTP.
Allowed: the numeric and ML libraries research already uses (numpy, scipy, scikit-learn, LightGBM) and other
local libraries that call no hosted model.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PRODUCTION = ("trading", "runner", "lab", "data", "infra", "monitoring", "scripts")
AI_PACKAGES = {
    "anthropic", "openai", "cohere", "mistralai", "ollama", "groq", "together", "replicate", "litellm", "vertexai",
    "langgraph", "langsmith", "llama_cpp", "transformers", "huggingface_hub", "sentence_transformers", "dspy",
    "instructor", "autogen", "autogen_agentchat", "crewai", "semantic_kernel", "haystack",
}
AI_PREFIXES = ("langchain", "llama_index")
AI_MODULES = ("google.generativeai", "google.genai", "google.ai.generativelanguage")
AI_HOSTS = ("api.openai.com", "api.anthropic.com", "generativelanguage.googleapis.com",
            "aiplatform.googleapis.com", "api.mistral.ai", "api.cohere.ai", "api.cohere.com", "api.groq.com",
            "openrouter.ai", "api.together.xyz", "api.replicate.com", "api-inference.huggingface.co",
            "api.deepseek.com", "localhost:11434", "127.0.0.1:11434")
DYNAMIC_IMPORTS = {"import_module", "__import__"}


def _is_ai(module: str) -> bool:
    top = module.split(".")[0]
    return top in AI_PACKAGES or top.startswith(AI_PREFIXES) or \
        any(module == m or module.startswith(m + ".") for m in AI_MODULES)


def _ai_uses(tree: ast.AST) -> list[int]:
    lines = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            hit = any(_is_ai(a.name) for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            hit = bool(base) and (_is_ai(base) or any(_is_ai(f"{base}.{a.name}") for a in node.names))
        elif isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else ""
            hit = name in DYNAMIC_IMPORTS and any(
                isinstance(a, ast.Constant) and isinstance(a.value, str) and _is_ai(a.value) for a in node.args)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            hit = any(host in node.value.lower() for host in AI_HOSTS)
        else:
            continue
        if hit:
            lines.add(node.lineno)
    return sorted(lines)


def test_no_production_module_uses_an_ai_sdk_or_ai_api():
    found = []
    for package in PRODUCTION:
        for path in sorted((ROOT / package).rglob("*.py")):
            found += [f"{path.relative_to(ROOT)}:{line}"
                      for line in _ai_uses(ast.parse(path.read_text(encoding="utf-8")))]
    assert found == []


def test_the_rule_detects_what_it_forbids():
    snippet = ("import numpy, sklearn.linear_model, lightgbm\n"
               "import anthropic\n"
               "from openai import OpenAI\n"
               "import langchain_core.messages as messages\n"
               "from google import genai\n"
               "import google.generativeai as gemini\n"
               "client = importlib.import_module('mistralai')\n"
               "local = __import__('ollama')\n"
               "URL = 'https://API.openai.com/v1/chat/completions'\n"
               "from llama_index.core import VectorStoreIndex\n"
               "from google.protobuf import message\n"
               "TELEGRAM = 'https://api.telegram.org/bot'\n")
    assert _ai_uses(ast.parse(snippet)) == [2, 3, 4, 5, 6, 7, 8, 9, 10]
