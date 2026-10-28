# PulseMorph Physical Agent

[Watch the 150-second English demo](https://youtu.be/9ue_CYnoW1w) · [Submitted demo assets](https://github.com/hik20/pulsemorph-physical-agent/releases/tag/v0.1.0-demo)

A general-purpose language model writes timed joint programs, tests a copied humanoid state in physics, reads measured feedback, then revises, commits or refuses. Music and expressive whole-body performance are the product direction. Short reaching and HOLD tasks are engineering probes.

This repository contains the **agentic module and minimal motion interfaces**. DanceModel architecture, training code, private datasets and weights remain private. Original agentic code is MIT; external robot, model and media rights are separate.

## What has actually worked

- The recorded music experiment: four live LLM calls, three 16.8-second physics previews, **zero applied dance commits**. The model refused the drifting dance. The video labels hypothetical physics separately from actual execution.
- A separate persistent-body episode: four live calls, two previews and two actual commits. Final wrist errors were 11.398/11.697 mm inside fixed 12 mm tolerances, with exact preview/commit numerical parity.
- Post-submission maintenance: zero-work rejected-action feedback and real 20 ms observation journals now support sampled HOLD verification. A new owned standing recipe avoids private source-motion initialization.
- A new isolated Windows Python environment installed 30 hash-verified cached public wheels, passed dependency checks and 33 targeted tests. The source-only CLI then completed a 2.4-second physical HOLD probe: 120 control calls, 2,400 physics steps, maximum wrist error 4.053 mm over a 0.4-second sampled hold. [Witness and limits](evidence/portable-firstuse-20261028.json). An earlier failed setup attempt exposed a default-vendor assumption and was preserved before the additive fix.

The maintenance witness used **zero live LLM calls**. Hosted-provider task mode is implemented and fixture-tested; this release does not claim a live hosted-model task, general reach, reliable musical dance, hardware control, continuous hold between samples or newly learned motor weights.

## Architecture

1. The model proposes typed, named-joint targets from the instruction, music description and body observation.
2. A local compiler checks continuous position, speed and acceleration bounds.
3. A complete copied physics/controller state previews the candidate; the actual parent remains unchanged.
4. Measured feedback returns to the model, which explicitly revises, commits an eligible preview or refuses.
5. Local execution checks observation identity, finite budgets and exact preview/commit trace agreement.

MuJoCo supplies dynamics. The current portable hybrid uses classical arm servos with the fixed pretrained G1 actor for legs/trunk. The LLM does not replace every low-level controller. Simulation time pauses while the model deliberates. The procedural 100 BPM score supplies a known construction grid; it is not evidence that an audio model understood arbitrary music.

## Setup and first local probe

The checked platform is Windows amd64 with CPython 3.11. A fresh isolated environment on the same host was tested; a clean operating system and fresh network asset acquisition were not.

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install --require-hashes -r experiments/agentic/app/bootstrap/requirements-win-cp311-public.lock
```

The 30-wheel recipe is pinned to public PyPI artifacts. Our installation witness used already downloaded, verified wheels offline. `requirements-observed.txt` is a direct-version summary, not a replacement for the full hash lock. The CLI runs directly from this source tree and does not need a private Codex installation or editable package install.

Model, body and mesh binaries are **not distributed here**. `asset_identity.json` identifies 70 required upstream files (69,182,643 materialized bytes), exact hashes, commit, newline conversion and license pointers. Follow [portable setup](design/agentic/BODY_AGENT_PORTABLE_V5.md) and review [external dependencies](docs/DEPENDENCIES.md). Retain upstream notices; do not deserialize untrusted motion pickles.

After obtaining the pinned public assets, materialize a separate runtime copy and create the owned profile:

```powershell
.venv/Scripts/python -I -B experiments/agentic/embodied/materialize_body_assets_v5.py --source ./upstream-protomotions --destination ./local/body-vendor-v5 --identity ./asset_identity.json
.venv/Scripts/python -I -B experiments/agentic/embodied/prepare_body_profile_v5.py --config ./body-config.json --build
```

The guide supplies `body-config.json` with explicit interpreter, vendor, profile and output paths, plus a model-free HOLD request. First prepare a bounded plan, review its inputs/budgets, then explicitly approve that exact digest:

```powershell
.venv/Scripts/python -I -B experiments/agentic/embodied/run_body_agent_v5.py --config ./body-config.json --request ./hold-request.json --output ./local/body-jobs-v5/hold_001 --plan ./hold-plan.json
Get-FileHash ./hold-plan.json -Algorithm SHA256
.venv/Scripts/python -I -B experiments/agentic/embodied/run_body_agent_v5.py --execute --plan ./hold-plan.json --approved-plan-sha256 <reviewed-sha256>
```

Preparation does not construct the physical body. Execution is exclusive, finite and has no automatic retry. Failures return a nonzero process exit. Actual and sandbox traces, failure counters, source pins and sampled observations are preserved. The additive reference adapter uses the configured profile and leaves the shared motion module unchanged.

Optional hosted task mode reads only `OPENROUTER_API_KEY` or `NVIDIA_API_KEY` from the private process environment. OpenRouter uses the free Nemotron profile without paid fallback. Requested/reported model identity is retained. Do not commit keys, local configs, profiles, jobs or authority files. Provider advice remains a proposal, never execution authority.

The optional V5 HTTP page is explicitly an **engineering setup dashboard**, with job POSTs disabled. It is not an empty dance studio advertised as ready. The historical V4 launcher and 8779 recorded-candidate viewer require their original prepared-host artifacts; they are not portable first-use commands. The final video is available at the link above.

## Validation and historical boundaries

Run the focused controller-free tests after installing the pinned dependencies:

```powershell
.venv/Scripts/python -m unittest discover -s tests -p "test_agentic_body_*v5.py"
.venv/Scripts/python -m unittest discover -s tests -p "test_agentic_portable_reference_v5.py"
.venv/Scripts/python -m unittest discover -s tests -p "test_agentic_embodied_core_v2.py"
```

Fixtures simulate state and providers. Only the separately recorded native witness is physical evidence. The public source tag `v0.1.0-demo` retains the original submitted code; maintenance added on 28 October must not be represented as pre-deadline evidence.

The portal showed **Submitted 1/1**; the final YouTube, GitHub and deck V3 links were verified after save/reload. No organizer-confirmed deadline extension or eligibility is claimed. A separate confirmation email was not found in the checked search.

## Sponsor contributions

- **Nemotron:** one actual later Nemotron 3.5 Super VL critique through OpenRouter reviewed two recorded images (17.687 seconds, 5,845 tokens) and recommended revision. Its advice was not motor-tested and it did not author the original dance.
- **Meterless:** earlier actual write/query/fresh-process reopen persistence was verified. The newest music experiment is not itself a retrieved-memory demonstration.
- **Zetaris:** after the initial submission, a real AWS Mini Trial engine returned two provenance-bound recorded-trial summary rows in 7.438 seconds; SQL was saved as `PulseMorph_recorded_trial_evidence`. These were CTE literal inputs, not a persisted datasource or body-planner history retrieval. The experiment's compute/network/storage were subsequently removed; local evidence remains. This is not a pre-deadline compliance claim.
- **SpaceX AI / SpaceXSI:** no implemented use claimed.

See [pre-existing work](docs/PREEXISTING_CODE.md) and [submission disclosure](docs/SUBMISSION.md). The final English video uses approved Gemini 3.8 Flash-Lite TTS/Kore narration. Editing recorded frames or narration did not create new physical success.

## License

Original agentic code and the included interoperability helpers are covered by [LICENSE](LICENSE). Third-party source, checkpoints, robot/mesh assets, source motion and media retain their separate terms. The private DanceModel learning core is excluded and is not licensed by this publication.
