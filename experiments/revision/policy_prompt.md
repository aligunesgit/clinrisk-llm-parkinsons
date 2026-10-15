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
