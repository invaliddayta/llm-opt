"""Fixed prompt set for speculative-decoding benchmarks.

Mix mirrors how the local model is used (Hermes agent: tools, code, mail/chat,
summaries) plus the standard DFlash eval domains (math, code, chat). Long-context
prompts are built from local source files so they are reproducible offline.
"""

from pathlib import Path

SHORT = {
    "code_rbtree": "Write a Python implementation of a red-black tree with insert and delete. Include type hints.",
    "code_rust_lru": "Implement an LRU cache in Rust with O(1) get and put. Explain the ownership choices briefly.",
    "code_bash": "Write a robust bash script that backs up a directory to a timestamped tar.zst archive, keeps the last 7 backups, and logs to syslog.",
    "code_sql": "Given tables orders(id, customer_id, total, created_at) and customers(id, name, country), write SQL that returns the top 3 customers by revenue per country for 2025, and explain the window functions used.",
    "code_fix": "This Python function is supposed to merge overlapping intervals but has bugs. Fix it and explain.\n\ndef merge(iv):\n    iv.sort()\n    out=[iv[0]]\n    for s,e in iv:\n        if s < out[-1][1]:\n            out[-1][1] = e\n        else:\n            out.append([s,e])\n    return out",
    "math_gsm": "A store sells pencils at 3 for $1.20 and pens at 2 for $3.50. Maria buys 15 pencils and 8 pens and pays with a $50 bill. How much change does she get? Show your work.",
    "math_proof": "Prove that the square root of 2 is irrational, then generalize the argument to show sqrt(p) is irrational for every prime p.",
    "math_prob": "Two fair dice are rolled until the sum is either 7 or 8. What is the probability that the final sum is 7? Explain step by step.",
    "chat_email": "Draft a polite but firm email to my landlord explaining that the heating has been broken for 9 days, that I reported it twice already, and that I expect a repair date within 48 hours.",
    "chat_plan": "I have 3 days in Lisbon in November with a moderate budget. Make a day-by-day plan with food recommendations and tips for avoiding crowds.",
    "chat_explain": "Explain how a transformer's KV cache works and why it speeds up autoregressive generation, for a software engineer who knows basic linear algebra.",
    "writing_story": "Write a 400 word story about a lighthouse keeper who discovers the light has been signalling to someone.",
    "writing_summary": "Summarize the causes, key events and consequences of the 2008 financial crisis in about 300 words, in plain language.",
    "tool_json": "You are an agent with tools get_weather(city: str, unit: 'c'|'f') and send_message(to: str, text: str). The user says: 'Tell Anna in a message whether she needs an umbrella in Berlin tomorrow.' Output ONLY the JSON array of tool calls you would make first, then explain your plan in one sentence.",
    "tool_yaml": "Convert this into a Kubernetes Deployment + Service YAML: app 'hermes-api', image ghcr.io/acme/hermes-api:1.4.2, 3 replicas, port 8080, liveness probe on /healthz, 256Mi memory limit, and a ClusterIP service on port 80.",
    "multilingual": "Translate into German and French, keeping the tone casual: 'Hey, I'm running a bit late, the train got stuck outside the station. Start without me, I'll be there in 20 minutes.'",
}

_ROOT = Path(__file__).resolve().parent.parent / "llama.cpp"


def _read(rel, max_chars):
    text = (_ROOT / rel).read_text(errors="replace")
    return text[:max_chars]


def long_prompts():
    return {
        "long_code_qa": "Here is a C++ source file:\n\n```cpp\n" + _read("common/speculative.cpp", 60000)
        + "\n```\n\nExplain how the DFlash drafting path in this file works, step by step, and point out any potential bugs.",
        "long_doc_summary": "Here is a README:\n\n" + _read("tools/server/README.md", 50000)
        + "\n\nWrite a concise summary of the server's main features and list every endpoint mentioned.",
    }


def all_prompts(include_long=True):
    p = dict(SHORT)
    if include_long:
        p.update(long_prompts())
    return p
