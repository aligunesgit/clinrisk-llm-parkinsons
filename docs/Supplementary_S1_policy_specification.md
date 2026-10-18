# Supplementary File S1 — ClinRisk-LLM safety specification and frozen policy

## S1.1 Provenance
The evaluated policy was generated on 23 August 2026. Its exact user prompt and returned JSON were recovered on 18 October 2026 from the contemporaneous local execution transcript. The recovered command used Claude Code CLI 2.1.241 without explicit model or system-prompt flags; the contemporaneous CLI banner reported Sonnet 5. The original prompt, verbatim response and provenance are supplied in Supplementary File S3 and `docs/original_policy_generation/`. Policy generation used Claude Code CLI v2.1.241 with its default system configuration; no custom system-prompt override was supplied. The archived materials document the experiment-specific instructions and the resulting policy used in the evaluation. Earlier statements that the original user prompt could not be provided are corrected by this recovery.

The specification in S1.3 is the later reconstruction used for all 40 revision generations; it differs from the recovered original wording. Those generations therefore do not isolate sampling variability under the original prompt.

## S1.2 Context variables
| Symbol | Key | Definition |
|---|---|---|
| $b_t$ | `budget` | fraction of the development pool already labelled |
| $\widehat{Sens}_t$, $\widehat{Spec}_t$ | `sens`, `spec` | three-fold out-of-fold sensitivity/specificity of a symmetric RBF-SVM fitted on the labelled set; `None` if either class has < 3 labels |
| $\pi_t$ | `prevalence` | fraction of PD among labelled observations |
| $m_t$ | `margin` | median absolute decision value on the unlabelled pool under the previous risk weight |
| $\bar p_t^-$ | `neg_frac` | fraction of unlabelled observations predicted negative |
| $g_t$ | `trend` | "declining" / "flat" / "improving": OOF sensitivity versus the previous step (±0.02) |

## S1.3 Safety specification (verbatim prompt used for regeneration)

You are designing a frozen calibration policy for cost-sensitive active learning on participant-level Parkinson's disease biosignals (an RBF-SVM classifier; class 1 = PD). The policy is generated once, stored, and replayed deterministically; it never sees test data, never labels anyone and never predicts any participant's diagnosis.

## Safety specification
1. Missed PD cases (false negatives) are more costly than false alarms, but specificity must be protected before sensitivity pressure is increased.
2. Escalate pressure towards sensitivity only when there is adequate specificity headroom and evidence that sensitivity is poor or declining.
3. If specificity collapses, immediately back off the risk weight and rely on targeted querying of possible false negatives instead.
4. Change actions gradually (avoid oscillation) and stop changing the risk weight late in the labelling budget so that the final model is stable.
5. Use only the allowed inputs below.

## Allowed inputs (dictionary `ctx`), computed only from the labelled development pool
- `budget`: fraction of the development pool already labelled (0-1).
- `sens`, `spec`: three-fold out-of-fold sensitivity and specificity of a symmetric SVM fitted on the labelled set; `None` when either labelled class has fewer than three observations (cold start).
- `prevalence`: fraction of PD among labelled observations.
- `margin`: median absolute SVM decision value on the unlabelled pool under the previous risk weight.
- `neg_frac`: fraction of unlabelled observations currently predicted negative.
- `trend`: one of "declining", "flat", "improving" (recent OOF sensitivity trend).

## Action space
- `r` (risk weight): positive-class weight of the SVM, integer in {1,2,3,4} (1 = symmetric training).
- `a` (negative-side query priority): integer in {1,2,3,4}; candidates predicted negative have their uncertainty score divided by `a`, so larger `a` prioritises possible false negatives for annotation.
Previous actions `prev_r`, `prev_a` are supplied; the loop starts at (1,1).

## Output
Return ONLY one Python code block defining

    def policy(ctx, prev_r, prev_a):
        ...
        return r, a

Use plain Python only: no imports, no attribute access, no I/O, no randomness. You may use `min`, `max`, `abs`, `int`, `round`. Both returned values must be integers in {1,2,3,4}. Choose concrete numeric thresholds yourself and include brief comments.


## S1.4 Frozen policy metadata (`experiments/clinrisk_llm_policy.json`)

```json
{
  "name": "ClinRisk-LLM distilled policy",
  "generator": "Claude Sonnet 5 via Claude Code CLI",
  "generated_on": "2026-08-23",
  "test_independent": true,
  "allowed_inputs": [
    "budget_fraction",
    "oof_sensitivity",
    "oof_specificity",
    "labeled_prevalence",
    "median_abs_margin",
    "negative_fraction_unlabeled",
    "oof_sensitivity_trend"
  ],
  "thresholds": {
    "sensitivity_very_low": 0.6,
    "sensitivity_low": 0.75,
    "specificity_critical": 0.5,
    "specificity_low": 0.6,
    "specificity_adequate": 0.7,
    "margin_low": 0.1,
    "negative_fraction_high": 0.85,
    "late_budget": 0.85
  },
  "defaults": {
    "risk_weight": 2,
    "negative_query_priority": 2
  },
  "summary": "Protect specificity first; escalate risk only with adequate specificity headroom and poor OOF sensitivity; use negative-side querying as the preferred targeted lever; smooth changes by one level and freeze risk late in the budget.",
  "generation_archive": "docs/original_policy_generation/",
  "original_cli_version": "2.1.241",
  "model_provenance": "Contemporaneous CLI banner reported Sonnet 5; no explicit --model argument or resolved API model identifier was recovered.",
  "user_prompt_recovered_on": "2026-10-18",
  "system_context_recovered": false
}

```

## S1.5 Frozen policy as executed (`experiments/clinrisk_agent_analysis.py`)

```python
def llm_policy(ctx, prev_r, prev_a):
    sens,spec=ctx["sens"],ctx["spec"]
    if sens is None or spec is None: nr,na=2,2
    elif spec<.50: return 1,4,"specificity_guard"
    else:
        if spec<.60: nr=min(prev_r,2)
        elif spec>=.70 and sens<.60: nr=4
        elif spec>=.70 and sens<.75: nr=3
        elif spec>=.70 and ctx["trend"]=="declining" and ctx["margin"]>=.10: nr=min(prev_r+1,4)
        else: nr=prev_r
        if spec<.60: na=3
        elif spec>=.70 and sens<.75 and ctx["neg_frac"]>=.85: na=3
        elif ctx["margin"]<.10 and spec>=.60: na=max(prev_a,3)
        elif ctx["prevalence"]<.15 or ctx["prevalence"]>.85: na=max(prev_a,2)
        else: na=prev_a
    nr,na=smooth(nr,prev_r),smooth(na,prev_a)
    if ctx["budget"]>=.85: nr=prev_r
    return nr,na,"llm_distilled"
```

## S1.6 Rationale and evidential status of the safeguards

| Component | Design rationale | Evidential status |
|---|---|---|
| FN:FP = 4:1 | Study an asymmetric missed-case preference under one shared evaluation loss | Experimental preference; not elicited clinical utility |
| Specificity guard and escalation conditions | Limit indiscriminate positive prediction while allowing sensitivity pressure | Methodological safeguards; not a clinical safety guarantee |
| Specificity 0.50/0.60/0.70 and sensitivity 0.60/0.75 | Implement ordered guard and escalation branches | Frozen implementation cutoffs; not clinically validated limits |
| Risk and query actions in {1,2,3,4} | Bound the policy's permitted adjustments | Restricted experimental action space |
| Margin 0.10, negative fraction 0.85, prevalence [0.15,0.85] | Define additional feedback conditions | Heuristic implementation choices; margin is not a calibrated probability |
| One-level changes and freeze at budget 0.85 | Reduce oscillation and late-stage variation | Stabilization heuristics; optimality not established |

The recovered original prompt specifies allowed inputs, actions, asymmetric loss, test independence and conservative escalation/de-escalation goals. The numerical cutoffs, defaults, smoothing and late-budget freeze appear in the recovered LLM response. The four clauses in S1.3 summarize the design during revision; the original wording is supplied separately in S3. No clinical derivation of these numerical cutoffs is claimed.

The OOF specificity guard uses a symmetric surrogate trained on actively selected labels. It neither enforces the specificity of the weighted classifier nor guarantees specificity on a new population. Static validity checks, specification consistency, clinical plausibility and demonstrated clinical safety are distinct forms of assessment. Independent clinical review has not been completed.
