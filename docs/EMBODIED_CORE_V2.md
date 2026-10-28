# Additive embodied core v2 — 28 October 2026

Status: implemented, fixture-tested, then exercised in a separately root-approved
native gate on 28 October 2026 after initial submission. The gate used a scripted
provider and genuine native body execution, with zero LLM service calls. It does
not revise the frozen competition evidence. Existing app entry points remain on
their frozen versions. See the dated witness below.

## Two corrected paths

`fork_v2.PreviewSession` supplies complete zero-work rejection proof around the
same pure action compiler, before fork allocation or preview budget reservation.
The 0.4-second / 0.35-radian candidate still rejects at 13.125 rad/s² against the
unchanged 12 rad/s² limit. Rejection extrema are normalized to strict JSON
primitives: NumPy scalar extrema otherwise pass JSON encoding but fail the
provider context primitive audit. The genuine violation now reaches the next
fixture model request, and the agent can propose a smaller candidate.

`runtime_v2.PersistentEpisode` records backend observations after each actual
20-ms control boundary. Revision remains action-level: intermediate samples have
the previous revision; the last completed sample advances it once. Task sample
identity is `(revision, control_index)`, implemented in additive `tasks_v2` and
`tasks_dual_v2`; repeated identical samples are idempotent and altered samples at
the same identity fail. Hold thresholds and maximum sample gaps are unchanged.

`fork_v2` binds preview and committed journals by SHA256 and exact equality.
Preview journal tampering is checked before actual commit work. The orchestrator
v2 fetches ordered journals through local accessors, verifies count, hash, every
20-ms time/index step, action revision and final endpoint, and folds the task
verifier across every real sample. Hypothetical verification never updates
actual task state. Missing samples fail closed; none are interpolated or invented.
Full journals are saved separately (maximum 50 samples; each finite observation
bounded to 512 KiB), so they do not inflate provider history or the single-receipt
limit. Receipts carry journal count/hash. No automatic candidate selection exists.

## Opt-in composition and live boundary

Compose only `runtime_v2.PersistentEpisode`, `fork_v2.PreviewSession`,
`preview_orchestrator_v2.PreviewOrchestrator` and the applicable v2 task verifier.
The v1 observation wire schema and fixed task DTO are retained; v2 task-state and
run schemas identify the new semantics. Do not pass a v1 task-state cache into v2.
No application entry point or authority profile has been automatically changed.

The existing physical backend already performs its FK inside `advance`.
Its `observe` reads those updated body positions and computes two wrist
Jacobians when uncached. Sampling each control therefore adds observation and
Jacobian work compared with action endpoints, although it adds no FK or physics
steps itself. A future live run must explicitly budget and measure those extra
Jacobian reads, storage and wall time under a new authority. The injected fixture
result does not admit that live run or establish motor performance.

## Validation

12 targeted tests passed in 1.769 seconds using the prepared app public Python
3.11 environment:

    python -m unittest discover -s tests -p test_agentic_embodied_core_v2.py -v

Coverage: real compile-limit feedback in the next fixture model context;
zero-work complete rejection proof; 20 ordered samples across 0.4 seconds;
positive 0.2-second sampled hold for single and dual wrists; endpoint-only v1
fails that same hold; hypothetical success remains hypothetical; gap and altered
sample rejection; pre-commit journal tamper rejection; missing sample rejection;
stale actions/commits, duplicate commit no-work, preview budgets and cached reads;
a journal above the old 512-KiB receipt limit without excess model context.
These fixtures simulate counters and state; they execute no physical backend.

`runtime_v1.py`, `fork_v1.py`, `preview_orchestrator_v1.py`, `tasks_v1.py` and
`tasks_dual_v1.py` were byte-compared with the public source snapshot and remain
identical. Competition tags, old source pins, media, logs, shared motion contracts,
private DanceModel learning code and model assets were not modified. GitHub
publication is pending root review.

## Post-submission native witness, 28 October 2026

Root executed the separately approved run-001 after code and authority review.
An independent audit then read only saved JSON/files and recomputed distances,
sequence bindings and hashes without importing a runtime or native module.
All 18 audit checks passed, including all 136 source pins unchanged.

The 20 actual committed observations span 0.39999999999995617 seconds, with a
maximum gap of 0.01999999999999802 seconds. Every sample remains within the
unchanged 12-mm tolerance. Maximum and final actual wrist error are both
3.4913661422448894 mm, meeting the fixed 0.2-second sampled-hold requirement.
Preview and actual sample journals are identical and hash-bound. The original
13.125 > 12 rad/s² compiler rejection used zero work and appears in the next
scripted provider request. Action revisions stay unchanged for 19 intermediate
samples and increment once at the final sample.

Aggregate native work: 140 online ORT calls plus one warmup, 2,800 physics steps,
141 FK and 282 wrist Jacobian calls. Parent work is 120 controls / 2,400 steps /
121 FK / 242 Jacobians; the sandbox suffix adds 20 / 400 / 20 / 40. One model,
one initialization and one ORT session are shared under the approved authority.
The three provider calls are explicitly scripted receipts, not live LLM calls.
This proves this bounded native sampled-hold and parity gate, not general motor
admission, continuous occupancy, autonomous reasoning performance or dance quality.

Saved-only audit:
C:/Users/halke/.cache/pulsemorph-agentic/core-v2-native-pilot/saved-only-independent-audit-v1.json

Run evidence:
C:/Users/halke/.cache/pulsemorph-agentic/core-v2-native-pilot/run-001

The pilot still uses the existing admitted private initialization profile.
Portable owned-profile composition requires a new pinned plan and authority;
no second native run is implied by this witness.
