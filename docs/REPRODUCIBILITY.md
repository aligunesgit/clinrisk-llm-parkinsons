# Reproduction and provenance of the revision

## Archived version and subsequent files

The base analysis tag is `ijms-revision-1`, resolving locally to `ece1b8dd6d8191d9676765c297ac44e57abcd221`. Before the present documentation update, the previous repository HEAD was `a9bc6e8c5cb72e4be293747004fd9324755ccad0`. The documentation and operating-point additions are versioned separately under `ijms-reviewer-updates-20261018`; the base analysis tag is unchanged. Inspect the exact checkout using:

```bash
git rev-parse HEAD
git rev-list -n 1 ijms-revision-1
git status --short
```

`docs/artifact_sha256.json` records SHA-256 digests for the two principal analysis scripts, frozen policy metadata, archived regeneration user prompt and 40 raw generation records. These are file identities, not clinical validation or independent confirmation of model provenance.

## What is and is not archived

- Original generation: frozen policy implementation and metadata are present; the original verbatim prompt is unavailable.
- Regeneration: `experiments/revision/policy_prompt.md` is the exact archived user prompt passed as an argument to `claude -p`. It is not a complete system prompt or transcript of CLI-injected context.
- Raw outputs: each `results/revision/regenerated_policies*/policy_*.json` contains response text, requested model identifier, reported model-usage identifiers and session identifier. Preserve these fields. These records do not independently authenticate the service or its historical configuration.
- No temperature or sampling seed is set in the generation script. Generation used independent CLI invocations, not a controlled study of alternative wording. The reconstructed wording is common to the regenerations and cannot recover the missing original wording.
- Evaluation: archived code is statically checked and replayed with fixed analysis seeds. Replaying frozen policies is reproducible in a compatible environment; new generation need not return the same policies.

Do not label a reconstructed user prompt as the original system prompt.

## Data and preprocessing

Follow the README data layout. Gait requires 165 valid participant recordings, including GaPt12–GaPt16, plus demographics. Verify the source checksum manifest. Speech uses the per-participant mean of exactly three recordings; there are no missing values to impute. The identifier and outcome are removed, leaving 753 predictors.

Standardization and bandwidth use the development pool, including its unlabeled features, without test data. This is pool-based unsupervised preprocessing rather than preprocessing restricted to the currently labeled subset. OOF monitoring computes the symmetric surrogate's bandwidth on its labeled subset. In the PCA sensitivity analysis, PCA and subsequent scaling also use the development pool only.

## Commands with explicit working directories

From the repository root, create the environment and place the datasets as documented in the README. The complete rerun is:

```bash
bash experiments/revision/rerun_all.sh
python3 experiments/revision/reviewer1_operating_sensitivity.py
```

Alternatively, run from the repository root, in order:

```bash
python3 experiments/revised_cal_analysis.py
python3 experiments/clinrisk_agent_analysis.py
python3 experiments/make_revised_figures.py
python3 experiments/make_clinrisk_figures.py
python3 experiments/revision/burden_dca.py
python3 experiments/revision/prevalence_matched.py
python3 experiments/revision/external_validation.py
python3 experiments/revision/reviewer2_analyses.py threshold
python3 experiments/revision/reviewer2_analyses.py kmeans
python3 experiments/revision/reviewer2_analyses.py oof
python3 experiments/revision/reviewer2_analyses.py pca
python3 experiments/revision/paired_differences.py
python3 experiments/revision/policy_regeneration.py evaluate
POLICY_MODEL=claude-opus-5-5 python3 experiments/revision/policy_regeneration.py evaluate
POLICY_MODEL=claude-haiku-5-5 python3 experiments/revision/policy_regeneration.py evaluate
python3 experiments/revision/make_revision_figures.py
python3 experiments/revision/make_supplements.py
python3 experiments/revision/reviewer1_operating_sensitivity.py
```

Generating fresh policies requires an authenticated external CLI and is optional. It is not necessary for replay. The `generate N` command makes new external model calls; it is not part of the reproduction commands above.

## Interpretation of comparisons

The component analysis uses kernel k-means and truncates its final acquisition batch to the requested label count. At half budget, targets are 58 gait and 88 speech labels. The adaptive loop acquires whole batches and records nominal budgets with a one-percentage-point tolerance. Saved half-budget records contain 60–61 gait and 87–91 speech labels. The adaptive k-means sensitivity uses its own RNG schedule for clustering. Consequently, replacing rank spreading with k-means does not make the two experimental loops identical. Compare policy effects within the same loop rather than subtracting costs across tables.

OOF signals assess a symmetric surrogate, not the weighted classifier. Cross-fitting does not remove active-selection bias. Budget-bin dispersion combines available recorded steps and splits; threshold agreement is descriptive, depends on event frequency and is not proof of a clinical safety guarantee.

Repeated outer splits overlap. Bootstrap intervals, win/tie proportions and percentile ranges summarize the observed resampling variation; they are not independent population replications or inferential significance tests. PCA sensitivity does not establish absence of overfitting. Independent speech data are a framework replication with grouped splitting, not transfer of the Sakar-trained model.

The new utility sensitivity analysis rescales existing errors; it does not retrain at each utility. Transported testing scenarios assume unchanged sensitivity/specificity. Additional total tests include additional true-positive and false-positive referrals; do not report additional false-positive tests as total additional investigations.

## Output mapping

| Reviewer concern | Script | Principal saved output |
|---|---|---|
| Threshold reference | `reviewer2_analyses.py threshold` | `results/revision/r2_threshold.json` |
| Diversity sensitivity | `reviewer2_analyses.py kmeans` | `results/revision/r2_kmeans.json` |
| OOF reliability | `reviewer2_analyses.py oof` | `results/revision/r2_oof.json` |
| Speech dimensionality | `reviewer2_analyses.py pca` | `results/revision/r2_pca.json` |
| Paired differences | `paired_differences.py` | `results/revision/paired_differences.json` |
| Policy replay | `policy_regeneration.py evaluate` | `results/revision/policy_regeneration*.json` |
| Clinical burden | `burden_dca.py` | `results/revision/burden_dca.json` |
| Fixed-prediction utility | `reviewer1_operating_sensitivity.py` | `results/revision/reviewer1_*` |

Split-level pickle files are created locally and excluded from version control. Regenerate them before scripts that depend on them. No independent clinician assessment has been completed; structured expert assessment is planned for future work.
