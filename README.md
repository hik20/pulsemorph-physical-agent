# PulseMorph Physical Agent

This public repository covers the agentic physical-feedback experiments only. DanceModel neural-network architecture, training code, datasets and weights remain private and are not licensed or distributed here. Shared motion contracts are included only to support the agentic runtime.

PulseMorph lets a language-model agent author timed movement programs, test them in a copied humanoid state, inspect physical feedback, and revise, choose or refuse execution. Dance is a visible stress test of the agent architecture. Interactive movement creators and robotics researchers are the intended users.

This is a **local research prototype**, not a claim of general robot control or a newly trained motor policy. LLM robotics has prior work; we do not claim to be its first demonstration. The contribution here is the explicit body-bound program/experiment/feedback/decision loop and its traceable execution boundaries.

## Demonstrated results

- A real music experiment used four live model calls and three full 16.8-second physics previews. The model wrote all 29 requested joint trajectories, revised them from measured feedback, and refused to commit the drifting run. The video correctly labels this as a **physics preview, not an applied dance**.
- A separate reaching task used four live calls, two previews and two actual commits on one persistent body. Final right/left wrist errors were 11.398/11.697 mm within fixed 12 mm tolerances. Each selected preview and actual commit matched the recorded full state and numerical execution trace.
- Classical contact/balance controller experiments are separate engineering work. Short standing/movement tests passed, but long music performance admission remains open. No general dance, hardware safety, real-time control or learned new weights are claimed.

See `docs/SUBMISSION.md`, `docs/PREEXISTING_CODE.md`, `docs/DEPENDENCIES.md` and the small `evidence/` summaries. A two-and-a-half-minute English demo video, English captions and seven-slide editable deck are supplied separately in the local submission package. Public URLs are not fabricated.

## Architecture

1. The model writes a typed movement program from the instruction, complete music summary, body state and joint limits.
2. A local compiler creates continuous reference curves and checks position, speed and acceleration limits.
3. A complete physics/controller state copy runs a hypothetical experiment. The actual parent does not move.
4. Measured movement, posture, contact and drift feedback returns to the model.
5. The model explicitly revises, commits an eligible preview, or refuses. Local execution authority remains separate from model proposals.

MuJoCo computes actual dynamics. Pretrained G1 tracking, classical servos, inverse kinematics and later contact-control experiments are identified separately. References, hypothetical results and realized body states are distinct records. Model thinking pauses simulation time. These experiments do not retrain motion-generator weights.

## Setup and run

Python 3.11 is the tested interpreter family. `requirements-observed.txt` records the tested direct numerical/media dependencies; it is not a complete transitive lock file.

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-observed.txt
.venv/Scripts/python -m pip install -e .
```

The physical runner additionally needs the pinned upstream ProtoMotions source, its G1 body/mesh/model/controller assets and the admitted safe initialization profile. These are **not in this source-only archive**. Obtain and review their separate licenses using `docs/DEPENDENCIES.md`; do not deserialize an untrusted motion pickle. The original prepared-host Windows scripts use the explicitly reviewed Python/profile under the user's cache. A fresh-machine asset acquisition/install has not been validated.

Validated prepared-host entry point:

```powershell
./experiments/agentic/embodied/start_body_agent_v4.ps1
```

This serves the bounded reaching/body-agent workflow on `http://127.0.0.1:8777/`. It does not claim the whole-body music goal is solved. The read-only music-candidate viewer is `src/pulsemorph/agentic/performance/review_server_v1.py`, port 8779, and additionally requires the exact local recorded media referenced in that module. It has no generation POST endpoint. The developmental 8778 music-generation application is kept unadmitted until its controller is qualified.

Controller-free tests can be run without starting a motor or model, once the numerical dependencies are present:

```powershell
.venv/Scripts/python -m unittest discover -s tests -p "test_agentic_performance_*.py"
```

Some tests import neighboring fixture modules from `tests`; retain the complete included fixture set. Fake/model-injected tests are not live model or physical success evidence.

## Model and sponsor configuration

- The recorded core experiments used the private local Codex bridge requesting `gpt-6.1-sol`; the bridge reports the requested identity and does not invent an unreturned model identity.
- Optional Nemotron adapters support NVIDIA NIM (`NVIDIA_API_KEY`) and OpenRouter (`OPENROUTER_API_KEY`). The OpenRouter adapter defaults to `nvidia/nemotron-3.5-super-vl:free`, with no paid fallback or retry. Configure secrets in a private environment, never in repository files. One real advisory review used `nvidia/nemotron-3.5-super-vl:free`:17.687seconds,2images and5845tokens; its decision was `revise`. The proposed movement was not executed or motor-validated. This is advisory model use, not a successful Nemotron-controlled dance.
- Meterless upstream write/query/fresh-process reopen was verified in earlier tasks. The newest dance branch is not itself a retrieved-memory demonstration.
- **Zetaris authenticated product use remains unverified.** An account was provisioned with a new email, but no authenticated product query was verified; no address or credentials are included. The event requires Zetaris and Meterless. This is an explicit submission-readiness blocker; a logo or fixture must not replace a real integration.

No hosted working-product URL is explicitly required by the checked public rules, but GitHub source with setup/usage, a 1–4 minute workflow video, a deck, sponsor explanation and pre-existing-code disclosure are required. The private submission form has not been inspected. See the official rules linked in `docs/SUBMISSION.md`.

## License and provenance

Original agentic project code in this repository is released under the MIT license in `LICENSE`. Shared motion-interface helpers are included for interoperability. Third-party code, model checkpoints, robot/mesh assets, source motion and generated media retain their separate licenses and rights. No ownership or license to those external assets is claimed. See `docs/PREEXISTING_CODE.md` and `docs/DEPENDENCIES.md`.
