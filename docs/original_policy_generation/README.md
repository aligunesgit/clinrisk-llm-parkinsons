# Recovery of the original policy generation

On 18 October 2026, the original user prompt and returned JSON policy were recovered from a local execution transcript dated 23 August 2026. Earlier revision documents incorrectly stated that the original prompt could not be provided: it had not been saved with the experiment artifacts, but remained in the execution transcript. The present recovery corrects that statement.

## Files and evidence

- `user_prompt.txt`: exact user-prompt argument of the successful `claude -p` command, with one terminal newline added for file storage.
- `response_verbatim.txt`: returned text, including its JSON fence, as captured in the successful command output.
- `response.json`: parsed JSON representation of that same response.
- `provenance.json`: source-file digest, original line numbers, UTC timestamps, invocation arguments and exported-file checksums. The source transcript remains local because it also contains unrelated work and authentication exchanges; only relevant scientific material is exported.

The successful invocation is recorded at 11:38:44 UTC on 23 August 2026 and its completed output at 11:40:06 UTC. An earlier attempt with the identical prompt returned a login error and is not the source of the policy. The local transcript records CLI version 2.1.241. No `--model`, `--system-prompt` or `--append-system-prompt` argument was supplied. The contemporaneous login banner reports Sonnet 5; the returned plain-text output does not independently identify the resolved API model.

## What recovery establishes

The prompt supplies the seven permitted context variables, action values 1–4, the 4*FN+FP evaluation preference, exclusion of held-out signals and conservative escalation/de-escalation goals. It requests an ordered deterministic JSON policy. The returned JSON supplies the numerical cutoffs, cold-start defaults, separate risk/query rules, smoothing and late-budget freeze. Thus, their source can now be traced to the actual generation rather than inferred from a reconstruction.

The evaluated Python `llm_policy` is a transcription of this output. Context keys are renamed by the implementation. Its emergency guard returns (1,4) immediately, including at late budgets, giving priority to the response's “overrides_all” guard. The ordinary smoothing and freeze are applied to non-emergency branches. The returned description also gives a generic postprocessing order; publishing both artifacts makes this precedence choice explicit. The scientific results refer to the executed Python implementation, not to an independently executed JSON interpreter. The JSON rationales are generated explanations, not validated clinical mechanisms or guarantees.

## Remaining boundary

This recovery supplies the original experiment-specific user prompt and output. It does not recover the effective Claude Code system messages, injected context, tool schemas or server configuration. The historical command relied on default CLI context. These missing items must not be represented by today's defaults or a newly written prompt.

The 40 revision policies were generated from `experiments/revision/policy_prompt.md`, a later reconstruction, not this recovered original wording. Their results compare policies generated from that documented reconstruction with the original frozen implementation. They do not isolate repeated sampling under an identical original prompt. No prediction or numerical result was changed by this recovery.
