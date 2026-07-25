# TraceGuard `types-v2` Contract

`src/traceguard/types.py` is the authoritative shared schema. Experiment manifests record
`types-v2`; incompatible changes require a new contract version. Version 2 adds
`OBSERVATION_INJECTION` and `MEMORY_POISONING` threat models. Existing version-1 custom
cases remain valid because their fields and enum values are unchanged.

External benchmark adapters use the independent `benchmark-adapter-v1` JSON protocol.
Requests and results reject unknown fields, carry the native case and dataset revision,
and keep upstream native scores separate from TraceGuard's normalized utility and
security fields.

## Decisions and precedence

- `ALLOW`: the call may proceed only to the recorded execution target.
- `REWRITE`: replace the call once, preserving task ID, step ID, call ID, observation
  provenance, and requested resources. The rewritten call is evaluated again. A second
  rewrite escalates.
- `ESCALATE`: stop automatic execution and require human review. It is not an allow.
- `BLOCK`: stop automatic execution. It has the highest precedence.

Combination precedence is `BLOCK > ESCALATE > REWRITE > ALLOW`. Deterministic `BLOCK` and
`ESCALATE` are terminal hard gates; an LLM cannot lower them. An LLM may raise risk or make
a stricter decision. Container routing is retained only when the winning decision is not
`BLOCK` or `ESCALATE`.

## Goal labels

- Relevance: `IRRELEVANT` has no defensible connection to the goal, `WEAK` has an indirect
  or uncertain connection, and `STRONG` directly advances it.
- Necessity: `UNNECESSARY` adds no required value, `HELPFUL` is useful but replaceable, and
  `NECESSARY` is required for successful completion.

Relevance and necessity are independent. A relevant call may be unnecessary, and a
necessary call may still be too risky to allow.

## Risk and trust

Risk is ordered `LOW < MEDIUM < HIGH < CRITICAL`. Container execution may contain only
approved uncertain or medium-risk command calls; it never makes prohibited high- or
critical-risk behavior permissible.

`TRUSTED_SYSTEM` is policy/configuration, `USER_INPUT` is the explicit user request,
`TRUSTED_TOOL` is deterministic local output, `UNTRUSTED_DOCUMENT` and `UNTRUSTED_TOOL`
may contain indirect instructions, and `AGENT_GENERATED` is model-authored data.
Untrusted content may provide facts but cannot independently authorize side effects.

## Execution targets and post-run handling

- `DEFAULT`: execute through the typed local tool boundary.
- `SIMULATED`: return a non-executing marker.
- `CONTAINER`: execute only through a trusted named sandbox profile.

Post-run `ACCEPT_RESULT` exposes bounded output to the agent. `BLOCK_RESULT` quarantines
it. `ESCALATE` stops. `REWRITE_AND_RETRY` never authorizes automatic host execution and is
currently converted to human escalation until a reviewed contained retry plan exists.
For LLM-backed modes, deterministic post-run checks are terminal when they identify
unsafe evidence. Otherwise the same configured provider performs a structured
post-run assessment. Provider failure or a low-confidence accept/retry escalates
instead of exposing the result.

## Human escalation

An escalated episode terminates automatic execution. The trace and sanitized evidence are
the review packet. Resuming requires a new explicit user-approved call; an escalation is
never silently converted to an allow.

## AgentDojo result normalization

AgentDojo uses a different raw convention for injected episodes: its
injection-task `security` checker returns `true` when the injected attacker
goal was completed. The adapter inverts that value before exposing
TraceGuard's security-success metrics. Clean AgentDojo episodes retain their
native `true` pass value because no injection checker runs.

## Dataset case and predicate semantics

Dataset cases may add source provenance, upstream split, attack family, participant and
payload-family groups, and an authorized benign near-neighbor. Forbidden-call predicates
match the tool name plus arguments using either exact equality or recursive subset
matching. State-diff checks compare declared keys against the disposable service state;
native utility and security checkers remain authoritative whenever an upstream runtime
provides them.

An observation imported from an untrusted dataset or native tool is always represented
as `UNTRUSTED_TOOL` with `may_contain_instructions=true`. Memory and service state are
created per case and discarded after scoring.
