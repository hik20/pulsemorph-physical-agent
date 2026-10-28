# PulseMorph — submitted evidence and post-submission maintenance

Checked 27 October 2026, approximately 23:16 UTC (28 October locally). This earlier rules review is retained as history. Current status: agentic-only source and demo published; portal Submitted1/1 and final video URL verified after reload. Eligibility/deadline extension remains unverified.

## Official requirements checked now

Primary source: [Open Agent Hackathon 2026 event and rules](https://hackathon.genai.works/event/open-agent-hackathon-2026), including the rule text embedded in the official page.

- Detailed rule 4 ends the build window **27 October 2026, 23:45 UTC**. The banner and key dates instead show 23:59 UTC. Use the earlier cutoff; this is an unresolved conflict on the same page, not an extension.
- Every submission must integrate **Zetaris and Meterless**. Cursor is optional; model/provider choice is open. NVIDIA contribution must be explained where applicable.
- Submit a **1–4 minute workflow video**, a deck, and a GitHub repository with source and setup/usage/dependency README. Include Colab links if used.
- Explain sponsor integration and its distinctive contribution. The deck covers users/problem, solution, workflow/architecture, stack, visuals and future work.
- Declare pre-existing product code. Existing libraries are allowed; new work is what the wildcard track scores.

## Submission text — ready to edit into the form

**Title:** PulseMorph: a body agent that tests before it acts

**Problem and users:** Interactive character and robotics builders need an agent that reasons about a real simulated body, rather than producing attractive motion descriptions that may be physically impossible.

**Solution:** PulseMorph connects a language model to typed motion tools, a persistent humanoid simulation, and explicit physical feedback. The model proposes joint motion, inspects hypothetical physics, and chooses whether to revise, commit, or stop. A commit is checked against the tested branch. Actual motion and hypothetical previews are kept separate, with bounded model and simulation budgets and durable evidence.

**Demonstrated workflow:** Existing recorded runs show live model-authored wrist actions, branch previews, commits, and a two-hand endpoint task on one persistent body. These are finite simulated demonstrations, not hardware safety or general skill mastery. The newer whole-body-to-music path is research in progress: an earlier model declined a drifting dance candidate, and the latest whole-body controller has a finite standing test. Do not present that standing test as a completed music-dance result. Use only the exact recorded outcomes visible in the final video.

**Architecture and contribution:** A language model chooses actions; versioned local tools enforce schemas, state hashes, budgets, numerical bounds and execution authority. MuJoCo supplies actual body/contact feedback. Pretrained motor components and newer classical control experiments are identified separately. No new skill training or shared model-weight learning is claimed.

**Memory:** Meterless is used as a real local persistent evidence store. Its upstream engine was tested by write, query and fresh-process reopen; later application records include measured trials and explicit rejections. This is symbolic evidence memory, not learned motor weights. The newest body-agent branch must not be described as using retrieved memory unless its own receipts show that.

**Sponsor evidence — post-submission:** An authenticated Zetaris SQL CTE returned two recorded-evidence rows in7.438seconds and was saved as `PulseMorph_recorded_trial_evidence`. It did not persist a source table or retrieve context for the body planner. This later evidence cannot establish pre-deadline compliance.

**Pre-existing components:** Declare the existing PulseMorph code (its exact pre-event dates are unknown and must be confirmed by the human participant; the workspace root has no Git history) and all reused libraries, G1/body assets, pretrained controller/model, source motions and their exact pinned origins. Separate new agent orchestration, physics evidence, application and controller experiments from pre-existing components. Repository availability does not establish redistribution rights for models, motions or character assets.

## Exact remaining inputs / review checklist

### Hosted product and NVIDIA prize clarification

The public submission rule text checked above specifies video, deck, GitHub and explanations; it does **not explicitly require a hosted working-product URL**. The private portal was inspected; it shows Submitted1/1 and the final video field was verified after reload. This does not establish deadline extension or organizer acceptance. Do not promise that a localhost URL is accessible to judges. A video/source submission and a publicly hosted interactive product are different deliverables.

The event lists **Best use of NVIDIA Nemotron**, awarding an NVIDIA DGX Spark Founder's Edition valued at $4,699. The public page does not name Nemotron3.5 Super VL as a mandatory version or publish further special-prize qualification tests. Existing NVIDIA-derived robot/model components are not automatically Nemotron use.

Configuration distinction: **Zetaris and Meterless are event requirements; NVIDIA NIM or OpenRouter is an optional model transport**, and Nemotron is relevant to the separate prize. [OpenRouter's model page](https://openrouter.ai/nvidia/nemotron-3.5-super-vl) and [explicit free variant](https://openrouter.ai/nvidia/nemotron-3.5-super-vl:free) were verified. The new OpenRouter adapter uses only `OPENROUTER_API_KEY`, defaults to that explicit `:free` model, and never falls back automatically to a paid model. Paid base-model selection is an explicit constructor option and still requires separate call authorization. One authorized live OpenRouter advisory review has now completed, as described below; no movement proposed by that review was executed. Free endpoint notices prohibit confidential/personal uploads and disclose provider logging; the intended inputs are our synthetic robot frames and bounded nonsecret measurements.

[NVIDIA's official model page](https://build.nvidia.com/nvidia/nemotron-3.5-super-vl-120b-a12b) documents model `nvidia/nemotron-3.5-super-vl-120b-a12b`, hosted API base `https://integrate.api.nvidia.com/v1`, text/image input, and a free prototype endpoint. A new local adapter supports a bounded visual-critic request with saved images and numerical feedback; the NVIDIA NIM adapter has fixture coverage, and the distinct OpenRouter transport completed one genuine advisory Nemotron review described below. **One live advisory contribution is claimed; no motor-validated improvement is claimed.** No download, new account or payment was initiated.

- [x] Post-submission authenticated Zetaris recorded-evidence CTE,2rows,7.438seconds; no planner/source-table integration claim.
- [x] Agentic-only repository https://github.com/hik20/pulsemorph-physical-agent ; final150-second English video https://youtu.be/9ue_CYnoW1w ; deckV3 uniqueassets in release v0.1.0-demo ; portal Submitted1/1. No eligibility or extension claim.
- [ ] Final 1–4 minute video: measured duration, understandable narration, legible actual/preview labels, no claim that an uncommitted preview was actual execution, and no music-dance success claim without a successful recorded full-window run.
- [ ] Deck and README tell the same current story and disclose local prepared-host dependencies, private asset/profile prerequisites, unresolved clean-machine setup and rights limitations.
- [ ] Confirm participant/team registration, selected track and pre-existing-code declaration. Do not invent registration or eligibility status.
- [ ] Verify final submission form against the official earlier deadline. No time-zone or calendar-date substitution.

Evidence pointers: `artifacts/runs/agentic/app_integrations_v1/manifest.json` records real Meterless persistence and missing Zetaris configuration; those flags are historical; the account is now provisioned but no authenticated query is verified. The body-agent delivery and individual run receipts remain the authority for numerical claims. This document performs no new model, controller, FK, simulation, upload or publication action.

## Actual Nemotron contribution — late update, 23:31 UTC

One explicitly authorized **live visual-critic call** completed through OpenRouter's free endpoint: requested and returned `nvidia/nemotron-3.5-super-vl:free`,17.687seconds,2existing robot images,5,845totaltokens, no tools or physics execution. Receipt and parsed response are in the private local evidence archive; a sanitized summary is included at `evidence/summary.json`. The critic recommended revision of the previously recorded hypothetical dance candidate. It identified limited phrase contrast and uncertain timing/contact evidence. This is an observed advisory multimodal contribution, **not the main choreography generator, new motor training, or motor-validated improvement**. This one call establishes advisory model use only; it does not establish a successful physical dance.

Critical interpretation: its statement that torque/contact data does not exist means that such data was not supplied to this critic; the simulator does record physical telemetry. Its description of negative world-X travel as sideways is axis-ambiguous and must not become an expert physical fact. Static sampled frames cannot prove continuous timing, support or safety. The model has no execution authority. A later authenticated Zetaris CTE query is verified; persisted data-source integration and body-planner retrieval remain unverified, and no retroactive competition-compliance claim is made.

## Zetaris provisioning update — 23:35 UTC

The user has located the official account-provisioning email. A cloud account is provisioned; login at [Zetaris Cloud](https://cloud.enterprise.zetaris.com/login) uses email OTP. **Successful login, an available execution engine and a live query are not yet verified.** Earlier missing-configuration flags remain historical process observations, not evidence that no account exists. No OTP, mailbox or credential store was accessed in this review.

Fastest legitimate path: sign in with the user's OTP, use an already available engine if present, run a read-only `SELECT 7 AS seven`, then retrieve an existing relevant telemetry/context table for real product use. A constant query establishes connectivity only, not meaningful integration. Do not create an AWS Mini cluster or accept billing without explicit cost authorization. If the account has no running engine, record that blocker.

Local deployment is not a deadline-sized fallback on this machine. Docker CLI29.4 is present but the daemon is unavailable (`Server=null`) and no DockerDesktop/backend process is running. Free disk was approximately1.75GiB onC and2.19GiB onD. Zetaris's [official installation test record](https://github.com/zetaris/getting-started/blob/main/docs/install/zetaris-installation-test-record.md) reports a roughly4GB first image download,6GB minimum/8GB recommended Docker memory, registry access and a supplied platform bundle. No install, image pull or paid resource was started.

The [official client HOWTO](https://github.com/zetaris/getting-started/blob/main/scripts/HOWTO.md) distinguishes UI-origin SQL proxy from direct REST/JDBC: a query requires authenticated authorization, organization identity, and optional engine selection. Cloud OTP authentication alone does not prove that our adapter's route is exposed by that deployment. Keep provisioning, authentication, execution and useful query evidence separate.

## Current delivery links — 28 October 2026

Submitted1/1 and finalvideoURL were verified after reload. Final video: https://youtu.be/9ue_CYnoW1w (150seconds, English Gemini3.8Flash-LiteTTS/Kore). Public source: https://github.com/hik20/pulsemorph-physical-agent ; release v0.1.0-demo contains unique deckV3 assets. Earlier provision/install observations above are dated history; one subsequent authenticated CTE returned2recorded rows in7.438seconds and was saved as PulseMorph_recorded_trial_evidence. No persisted source table/body-planner retrieval or pre-deadline sponsor compliance is inferred.
