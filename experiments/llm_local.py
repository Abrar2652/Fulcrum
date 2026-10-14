"""Plain-LLM baseline for H1 using a LOCAL open-source instruct model (Phase 6).

Local, reproducible, free: loads an instruction-tuned model from the shared
hf_cache and runs the same few-shot conflict-resolution task the calculus is fit
on, head-to-head on the SAME test conflicts. Greedy decoding for determinism.

Usage: python experiments/llm_local.py [model_key]
"""
from __future__ import annotations

import json
import os
import re
import sys

# Uses your standard Hugging Face cache (set HF_HOME to point elsewhere).
# Set FULCRUM_OFFLINE=1 to force offline loading from an already-downloaded cache.
if os.environ.get("FULCRUM_OFFLINE") == "1":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from experiments.run import fit_context_calculus  # noqa: E402
from experiments.synthetic import N_HEADS, make_dataset, split  # noqa: E402

MODELS = {
    "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen3b": "Qwen/Qwen2.5-3B-Instruct",
    "llama8b": "meta-llama/Llama-3.1-8B-Instruct",
}
N_TEST = 40
N_SHOTS = 24


def _fmt(c):
    return f"a={list(c.va)} b={list(c.vb)} ctx={c.ctx}"


def _messages(shots, tests):
    ex = "\n".join(f"{_fmt(c)} -> {'a' if c.winner == 1 else 'b'}" for c in shots)
    qs = "\n".join(f"{i}: {_fmt(c)}" for i, c in enumerate(tests))
    user = (
        "Each line is a conflict between evidence a and b (4 head-scores each) with a "
        "context tag. Exactly one of a/b wins by a fixed hidden rule that can depend on "
        f"the context tag. Learn it from these examples:\n\n{ex}\n\n"
        "Predict the winner for each item below. Output ONLY a JSON object mapping the "
        'index to "a" or "b", e.g. {"0":"a","1":"b"}. No explanation.\n\n' + qs
    )
    return [{"role": "system", "content": "You are a precise pattern learner. Output only JSON."},
            {"role": "user", "content": user}]


def run(model_key="qwen7b"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    data, _ = make_dataset(1000, seed=0, regime="context", n_ctx=N_HEADS)
    tr, te = split(data, 0.6, seed=1)
    shots, tests = tr[:N_SHOTS], te[:N_TEST]

    calc_pred, _, _ = fit_context_calculus(tr)
    calc_acc = sum(calc_pred(c) == c.winner for c in tests) / len(tests)

    name = MODELS[model_key]
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16,
                                                 device_map="cuda:0")
    prompt = tok.apply_chat_template(_messages(shots, tests), tokenize=False,
                                     add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=800, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    m = re.search(r"\{.*\}", text, re.S)
    pred = {}
    if m:
        try:
            pred = json.loads(m.group(0))
        except json.JSONDecodeError:
            pred = dict(re.findall(r'"(\d+)"\s*:\s*"([ab])"', m.group(0)))
    correct = sum(1 for i, c in enumerate(tests)
                  if str(pred.get(str(i), "?")).lower() == ("a" if c.winner == 1 else "b"))
    llm_acc = correct / len(tests)
    return {"model": name, "calc_acc": calc_acc, "llm_acc": llm_acc,
            "parsed": len(pred), "n_test": len(tests)}


def main():
    key = sys.argv[1] if len(sys.argv) > 1 else "qwen7b"
    r = run(key)
    print("=" * 56)
    print(f"H1 plain-LLM baseline (LOCAL {r['model']}, {N_SHOTS}-shot, n_test={r['n_test']})")
    print("=" * 56)
    print(f"  fulcrum calculus (fitted): acc = {r['calc_acc']:.3f}")
    print(f"  plain LLM (few-shot):     acc = {r['llm_acc']:.3f}  (parsed {r['parsed']}/{r['n_test']})")
    print("\n(The LLM must in-context-learn an arbitrary context->head rule; the calculus "
          "models it explicitly. NL benchmarks like BoardGameQA remain the primary LLM "
          "comparison and are deferred.)")


if __name__ == "__main__":
    main()
