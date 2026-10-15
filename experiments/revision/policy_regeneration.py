#!/usr/bin/env python3
"""Reviewer comment 1: sensitivity of ClinRisk-LLM to LLM generation randomness.

The safety specification (policy_prompt.md) is sent N times, in independent
sessions, to the same model used originally (claude-sonnet-5 via the Claude Code
CLI). Each response must be a pure Python function policy(ctx, prev_r, prev_a).
Responses are archived verbatim, statically checked (AST whitelist), and each
valid policy is replayed through the unchanged adaptive loop on all 60 splits.

Usage: [POLICY_MODEL=claude-opus-5-5] python3 policy_regeneration.py generate N   # calls the LLM
       python3 policy_regeneration.py evaluate     # replays archived policies
"""
from __future__ import annotations

import ast
import builtins
import json
import os
import re
import subprocess
import sys
from multiprocessing import Pool

import numpy as np
from sklearn.preprocessing import StandardScaler

from rev_common import OUT, ca, load_gait, load_speech, mean_ci, rc

# POLICY_MODEL selects an alternative LLM configuration (reviewer 2); the default
# reproduces the original generator and keeps the original output folder.
MODEL = os.environ.get("POLICY_MODEL", "claude-sonnet-5")
TAG = "" if MODEL == "claude-sonnet-5" else "_" + MODEL.replace("claude-", "").replace("-", "")
PROMPT = (rc.ROOT / "experiments" / "revision" / "policy_prompt.md").read_text()
GEN_DIR = OUT / f"regenerated_policies{TAG}"
GEN_DIR.mkdir(parents=True, exist_ok=True)
ALLOWED_CALLS = {"min", "max", "abs", "int", "round"}
CTX_KEYS = ("budget", "sens", "spec", "prevalence", "margin", "neg_frac", "trend")


def generate(n):
    for i in range(n):
        path = GEN_DIR / f"policy_{i:02d}.json"
        if path.exists():
            continue
        res = subprocess.run(["claude", "-p", PROMPT, "--model", MODEL, "--output-format", "json"],
                             capture_output=True, text=True, timeout=600, cwd=GEN_DIR)
        out = json.loads(res.stdout)
        json.dump({"index": i, "model": MODEL, "session_id": out.get("session_id"),
                   "model_usage": list(out.get("modelUsage", {}).keys()),
                   "response": out.get("result", "")}, open(path, "w"), indent=2)
        print("generated", path.name, flush=True)


def extract(text):
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.S)
    return m.group(1) if m else text


def check(code):
    tree = ast.parse(code)
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(funcs) != 1 or funcs[0].name != "policy":
        raise ValueError("expected exactly one top-level def policy")
    def is_ctx_get(n):  # harmless dict lookup; the only attribute access tolerated
        return isinstance(n, ast.Attribute) and n.attr == "get" and isinstance(n.value, ast.Name) and n.value.id == "ctx"

    # helper functions the model defines inside its own code may also be called
    local_defs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and not is_ctx_get(node):
            raise ValueError("disallowed attribute access")
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Lambda, ast.Global,
                             ast.Nonlocal, ast.With, ast.Try, ast.While, ast.ClassDef)):
            raise ValueError(f"disallowed node {type(node).__name__}")
        if isinstance(node, ast.Call):
            if not ((isinstance(node.func, ast.Name) and node.func.id in ALLOWED_CALLS | local_defs)
                    or is_ctx_get(node.func)):
                raise ValueError("disallowed call")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("dunder name")
    return tree


def compile_policy(code):
    check(code)
    env = {"__builtins__": {k: getattr(builtins, k) for k in ALLOWED_CALLS}}
    exec(compile(code, "<policy>", "exec"), env)
    fn = env["policy"]

    def wrapped(ctx, prev_r, prev_a):
        r, a = fn({k: ctx[k] for k in CTX_KEYS}, prev_r, prev_a)
        r, a = int(r), int(a)
        if r not in (1, 2, 3, 4) or a not in (1, 2, 3, 4):
            raise ValueError(f"out-of-range action {(r, a)}")
        return r, a, "regenerated"
    return wrapped


def load_policies():
    out = {}
    for path in sorted(GEN_DIR.glob("policy_*.json")):
        rec = json.load(open(path))
        try:
            out[path.stem] = (extract(rec["response"]), None)
            compile_policy(out[path.stem][0])
        except Exception as e:  # noqa: BLE001 - record and report invalid generations
            out[path.stem] = (None, str(e))
    return out


def job(args):
    name, code, dataset, seed = args
    if not DATA:  # macOS spawns fresh workers; load data once per worker
        DATA["gait"] = load_gait()[:2]
        DATA["speech"] = load_speech()
    X, y = DATA[dataset]
    pool, test = rc.split_indices(y, seed)
    sc = StandardScaler().fit(X[pool])
    Xp, Xt = sc.transform(X[pool]), sc.transform(X[test])
    ca.llm_policy = compile_policy(code)  # run_one dispatches non-baseline modes to ca.llm_policy
    try:
        rows = ca.run_one(Xp, y[pool], Xt, y[test], seed, "clinrisk_llm")
        r = next(z for z in rows if z["budget"] == 50)
        return {"policy": name, "dataset": dataset, "seed": seed, "ok": True,
                **{k: r[k] for k in ("cost4", "fnr", "specificity", "balanced_accuracy", "risk_weight")}}
    except Exception as e:  # noqa: BLE001
        return {"policy": name, "dataset": dataset, "seed": seed, "ok": False, "error": str(e)}


DATA = {}


def evaluate():
    DATA["gait"] = load_gait()[:2]
    DATA["speech"] = load_speech()
    pols = load_policies()
    valid = {k: c for k, (c, err) in pols.items() if c is not None}
    jobs = [(k, c, d, s) for k, c in valid.items() for d in DATA for s in range(rc.N_SPLITS)]
    with Pool(8) as p:
        rows = p.map(job, jobs)
    summary = {"n_generated": len(pols), "n_valid": len(valid),
               "invalid": {k: err for k, (c, err) in pols.items() if c is None}, "per_policy": {}}
    for k in valid:
        summary["per_policy"][k] = {}
        for d in DATA:
            rr = [r for r in rows if r["policy"] == k and r["dataset"] == d and r["ok"]]
            summary["per_policy"][k][d] = {m: float(np.mean([r[m] for r in rr])) for m in
                                           ("cost4", "fnr", "specificity", "balanced_accuracy", "risk_weight")}
            summary["per_policy"][k][d]["n_failed_runs"] = rc.N_SPLITS - len(rr)
    summary["across_policies"] = {}
    for d in DATA:
        costs = [summary["per_policy"][k][d]["cost4"] for k in valid]
        summary["across_policies"][d] = {
            m: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                "min": float(np.min(v)), "max": float(np.max(v))}
            for m, v in ((m, [summary["per_policy"][k][d][m] for k in valid])
                         for m in ("cost4", "fnr", "specificity", "balanced_accuracy", "risk_weight"))}
        summary["across_policies"][d]["cost4_ci_of_mean"] = mean_ci(costs)
    json.dump(summary, open(OUT / f"policy_regeneration{TAG}.json", "w"), indent=2)
    print(json.dumps(summary["across_policies"], indent=1))
    print("invalid:", summary["invalid"])


if __name__ == "__main__":
    if sys.argv[1] == "generate":
        generate(int(sys.argv[2]))
    else:
        evaluate()
