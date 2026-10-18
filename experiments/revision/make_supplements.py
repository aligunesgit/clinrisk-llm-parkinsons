#!/usr/bin/env python3
"""Build Supplementary Files S1 (specification + frozen policy) and S2 (regenerated policies)."""
from __future__ import annotations

import json
import re
from pathlib import Path

from rev_common import EXP, OUT, ROOT

DEST = ROOT / "manuscripts" / "IJMS" / "revision"
DEST = DEST if DEST.exists() else ROOT / "docs"
DEST.mkdir(parents=True, exist_ok=True)
MODELS = (("claude-sonnet-5", ""), ("claude-opus-5-5", "_opus55"), ("claude-haiku-5-5", "_haiku55"))


def s1():
    spec = (EXP / "revision" / "policy_prompt.md").read_text()
    frozen = (EXP / "clinrisk_llm_policy.json").read_text()
    src = (EXP / "clinrisk_agent_analysis.py").read_text()
    fn = src[src.index("def llm_policy"):src.index("def rule_policy")].rstrip()
    text = f"""# Supplementary File S1 — ClinRisk-LLM safety specification and frozen policy

## S1.1 Provenance
The evaluated policy was generated on 23 August 2026. Its exact user prompt and returned JSON were recovered on 18 October 2026 from the contemporaneous local execution transcript. The recovered command used Claude Code CLI 2.1.241 without explicit model or system-prompt flags; the contemporaneous CLI banner reported Sonnet 5. The original prompt, verbatim response and provenance are supplied in Supplementary File S3 and `docs/original_policy_generation/`. The effective CLI system context was not recovered. Earlier statements that the original user prompt could not be provided are corrected by this recovery.

The specification in S1.3 is the later reconstruction used for all 40 revision generations; it differs from the recovered original wording. Those generations therefore do not isolate sampling variability under the original prompt.

## S1.2 Context variables
| Symbol | Key | Definition |
|---|---|---|
| $b_t$ | `budget` | fraction of the development pool already labelled |
| $\\widehat{{Sens}}_t$, $\\widehat{{Spec}}_t$ | `sens`, `spec` | three-fold out-of-fold sensitivity/specificity of a symmetric RBF-SVM fitted on the labelled set; `None` if either class has < 3 labels |
| $\\pi_t$ | `prevalence` | fraction of PD among labelled observations |
| $m_t$ | `margin` | median absolute decision value on the unlabelled pool under the previous risk weight |
| $\\bar p_t^-$ | `neg_frac` | fraction of unlabelled observations predicted negative |
| $g_t$ | `trend` | "declining" / "flat" / "improving": OOF sensitivity versus the previous step (±0.02) |

## S1.3 Safety specification (verbatim prompt used for regeneration)

{spec}

## S1.4 Frozen policy metadata (`experiments/clinrisk_llm_policy.json`)

```json
{frozen}
```

## S1.5 Frozen policy as executed (`experiments/clinrisk_agent_analysis.py`)

```python
{fn}
```
"""
    text += "\n## S1.6 Rationale and evidential status of the safeguards\n\n| Component | Design rationale | Evidential status |\n|---|---|---|\n| FN:FP = 4:1 | Study an asymmetric missed-case preference under one shared evaluation loss | Experimental preference; not elicited clinical utility |\n| Specificity guard and escalation conditions | Limit indiscriminate positive prediction while allowing sensitivity pressure | Methodological safeguards; not a clinical safety guarantee |\n| Specificity 0.50/0.60/0.70 and sensitivity 0.60/0.75 | Implement ordered guard and escalation branches | Frozen implementation cutoffs; not clinically validated limits |\n| Risk and query actions in {1,2,3,4} | Bound the policy's permitted adjustments | Restricted experimental action space |\n| Margin 0.10, negative fraction 0.85, prevalence [0.15,0.85] | Define additional feedback conditions | Heuristic implementation choices; margin is not a calibrated probability |\n| One-level changes and freeze at budget 0.85 | Reduce oscillation and late-stage variation | Stabilization heuristics; optimality not established |\n\nThe recovered original prompt specifies allowed inputs, actions, asymmetric loss, test independence and conservative escalation/de-escalation goals. The numerical cutoffs, defaults, smoothing and late-budget freeze appear in the recovered LLM response. The four clauses in S1.3 summarize the design during revision; the original wording is supplied separately in S3. No clinical derivation of these numerical cutoffs is claimed.\n\nThe OOF specificity guard uses a symmetric surrogate trained on actively selected labels. It neither enforces the specificity of the weighted classifier nor guarantees specificity on a new population. Static validity checks, specification consistency, clinical plausibility and demonstrated clinical safety are distinct forms of assessment. Independent clinical review has not been completed.\n"
    (DEST / "Supplementary_S1_policy_specification.md").write_text(text)


def original_row():
    """Frozen-policy results read from the adaptive-policy summary (never hard-coded)."""
    o = json.load(open(ROOT / "results" / "clinrisk_agent" / "summary.json"))
    g, s = o["gait"]["clinrisk_llm"], o["speech"]["clinrisk_llm"]
    return (f"| claude-sonnet-5 (Aug 2026) | original frozen | {g['cost4']['mean']:.2f} | {g['fnr']['mean']:.3f} | "
            f"{g['specificity']['mean']:.3f} | {g['risk_weight']['mean']:.2f} | {s['cost4']['mean']:.2f} | "
            f"{s['risk_weight']['mean']:.2f} |")


def s2():
    lines = ["# Supplementary File S2 — Regenerated ClinRisk-LLM policies", "",
             "Each policy was generated in an independent session from the specification in File S1 through the "
             "Claude Code CLI (October 2026). It was statically checked and replayed unchanged on all 60 splits. "
             "Values are means at 50% labelling.", "",
             "| Model | Policy | Gait $C_4$ | Gait FNR | Gait specificity | Gait mean $r_t$ | Speech $C_4$ | Speech mean $r_t$ |",
             "|---|---|---|---|---|---|---|---|",
             original_row()]
    code = []
    for model, tag in MODELS:
        res = json.load(open(OUT / f"policy_regeneration{tag}.json"))
        for k, v in res["per_policy"].items():
            g, s = v["gait"], v["speech"]
            lines.append(f"| {model} | {k} | {g['cost4']:.2f} | {g['fnr']:.3f} | {g['specificity']:.3f} | "
                         f"{g['risk_weight']:.2f} | {s['cost4']:.2f} | {s['risk_weight']:.2f} |")
        for f in sorted((OUT / f"regenerated_policies{tag}").glob("policy_*.json")):
            r = json.load(open(f))
            m = re.search(r"```(?:python)?\n(.*?)```", r["response"], re.S)
            body = m.group(1).rstrip() if m else r["response"]
            code.append(f"### {model} — {f.stem} (session {r['session_id']})\n\n```python\n{body}\n```\n")
    text = "\n".join(lines) + "\n\n## Policy code (verbatim model output)\n\n" + "\n".join(code)
    (DEST / "Supplementary_S2_regenerated_policies.md").write_text(text)


if __name__ == "__main__":
    s1()
    s2()
    print("supplements written to", DEST)
