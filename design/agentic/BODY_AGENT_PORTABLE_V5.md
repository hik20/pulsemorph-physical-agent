# Portable V5 setup and bounded engineering probe

## 28 October update: fresh isolated installation and CLI HOLD passed

A new isolated CPython 3.11 environment on a separate Windows drive installed the
30 hash-verified cached public wheels without downloading anything. Dependency
checks and 33 targeted tests passed from a separate public-only source snapshot.
The CLI then completed its explicit finite 2.4-second native HOLD: 120 online
ONNX calls plus one warmup, 2,400 physics steps, 121 runtime FK and 242 wrist
Jacobian calls. Maximum wrist error during the 0.4-second sampled hold was
4.052848 mm; the fixed tolerance was 12 mm and the criterion 0.2 seconds.
No live language-model service was called. This is same-host isolated-install
and numerical HOLD evidence, not a clean operating-system/network-acquisition,
general reach, hosted-provider task, continuous hold or musical dance claim.

The first isolated CLI attempt failed before online control because the shared
motion assembler read its default checkout instead of the configured profile.
Its one warmup/one initialization FK/two Jacobian reads and zero control/physics
steps remain recorded. The correction is additive: `g1_portable_v5.py`,
`reference_portable_v5.py` and `runtime_portable_v5.py` pass the explicit profile.
The shared motion code and frozen V2 control state machine were not changed.
Regressions prove the absent-default case, exact configured identity rejection
and unchanged execution AST except for the reference import.

The setup HTTP page remains read-only. A short numerical probe does not admit
unbounded jobs. Every CLI execution still needs a newly prepared, reviewed exact
plan digest. Failure returns a nonzero exit. Actual and sandbox exports are
attempted independently, and closing source-pin changes invalidate the result.

## Obtaining only the required public payloads

The following is an acquisition recipe; its network fetch was not exercised by
the cached-asset installation witness. Review upstream notices before fetching
or distributing models and meshes. Git LFS is required for tracked payloads.

```powershell
$bodyAssetPaths = (Get-Content ./asset_identity.json -Raw | ConvertFrom-Json).files.path
$bodyAssetInclude = $bodyAssetPaths -join ','
$env:GIT_LFS_SKIP_SMUDGE = '1'
git clone --filter=blob:none --no-checkout https://github.com/NVlabs/ProtoMotions upstream-protomotions
git -C upstream-protomotions fetch --depth 1 origin f69fdcaabff31846b4d2247207417b57dc788828
git -C upstream-protomotions checkout --detach f69fdcaabff31846b4d2247207417b57dc788828
git -C upstream-protomotions lfs fetch origin f69fdcaabff31846b4d2247207417b57dc788828 --include="$bodyAssetInclude" --exclude=""
git -C upstream-protomotions lfs checkout $bodyAssetPaths
Remove-Item Env:\GIT_LFS_SKIP_SMUDGE
```

The offline materializer below verifies all 70 tracked blob identities and actual
payload hashes, rejecting unresolved pointers. It writes a new runtime copy and
never changes the source checkout. Keep the source's LICENSE/model notices and
Unitree asset notices separately; the runtime closure is not a license bundle.

## Preparation protocol and historical scope

V5 is additive. V1–V4 and their histories are unchanged. It can display the existing UI with `ready:false` without private profiles. Creating jobs remains explicitly disabled until a new owned initialization has passed a separately authorized native HOLD/reach gate and the core-V2 worker composition is reviewed. This is not yet a portable execution-success claim.

Use any selected CPython3.11 environment with the exact observed dependencies; the launcher has no developer-cache interpreter requirement. No installs occur automatically. `asset_identity.json` records70 upstream files: compiled ONNX, controller YAML, MJCF,63meshes and4deployment helpers, with raw SHA256, tracked-blob identity, newline materialization and license pointers. It contains no private paths or original motion clips.

Create a local config (do not commit host-specific paths):

```json
{
  "schema": "pulsemorph.body_config.v5",
  "runtime_python": ".venv/Scripts/python.exe",
  "vendor_root": "research/vendor/ProtoMotions",
  "profile_dir": "local/body-profile-v5",
  "artifact_root": "local/body-jobs-v5",
  "asset_identity": "asset_identity.json",
  "port": 8780
}
```

Paths resolve against this config file, not current shell directory. Asset-identity bytes are pinned in `config_v5.py`. Unknown/duplicate fields, linked paths, changed assets, mismatched versions and vendor-contained artifact roots are rejected. A per-install environment probe records interpreter and native-library hashes; runtime loaded-module origins must stay inside the selected environment. Different interpreter binary identity is not silently represented as the original host.

```powershell
& .venv/Scripts/python.exe -I -B experiments/agentic/embodied/prepare_body_profile_v5.py --config body-config.json
# After acquisition/materialization has passed, build an exclusive new profile:
& .venv/Scripts/python.exe -I -B experiments/agentic/embodied/prepare_body_profile_v5.py --config body-config.json --build
& ./experiments/agentic/embodied/start_body_agent_v5.ps1 -Config ./body-config.json -RuntimePython ./.venv/Scripts/python.exe
```

The default profile-preparation command remains read-only. A separate explicit offline materializer now copies verified assets into a **new disjoint destination** and deterministically converts only declared LF bytes. It checks the source Git commit and tracked blob SHA before publication, then the canonical payload SHA/size. Existing canonical bytes are accepted; unknown bytes, unresolved LFS payloads, source/destination overlap, links and existing destinations are rejected. There is no overwrite/resume/delete fallback; partial failures remain for review.

```powershell
# Acquisition is a separate user-reviewed step, NOT performed automatically:
# git clone --no-checkout https://github.com/NVlabs/ProtoMotions upstream-protomotions
# git -C upstream-protomotions config core.autocrlf false
# git -C upstream-protomotions checkout f69fdcaabff31846b4d2247207417b57dc788828
# Use GIT_LFS_SKIP_SMUDGE=1 until reviewing only required model/mesh payload sizes.
# Retain LICENSE.md, model LICENSE.pdf/NOTICE.txt/MODEL_CARD.md and Unitree notices
# in that acquired source; distribution is a separate decision.
& .venv/Scripts/python.exe -I -B experiments/agentic/embodied/materialize_body_assets_v5.py --source ./upstream-protomotions --destination ./local/body-vendor-v5 --identity ./asset_identity.json
# Set config.vendor_root to local/body-vendor-v5; then run profile preparation above.
```

Do not normalize the frozen project vendor. No full-checkpoint download is necessary for inference; inspect LFS pointers before downloading only required compiled-model/mesh blobs. Current source performs no network acquisition. Missing payloads must be supplied through a separately reviewed upstream acquisition step. The70-file materialization is a runtime closure, not a license-complete redistributed vendor package; retain the separately acquired source and notices.

Profile builder writes only safe NPZ/JSON, no pickle: owned symmetric standing q (hip_pitch−.15,knee+.30,ankle_pitch−.15,elbow+.30,other joints0),rootZ.79,identityWXYZ orientation,zero qvel. It validates named XML ranges. Reload recomputes the exact recipe, so editing both NPZ and its self-hash cannot admit an arbitrary pose. NPZ compressed size≤32KiB, exactly4 members and≤4KiB per decompressed array are checked before loading. These are an explicit proposed initialization, not FK-derived body locations or a balanced-pose guarantee. Its new backend uses existing unchanged advance/observe/closure/inference method bodies, while replacing private PhraseAssets/environment initialization with public profile/config verification. All native construction remains execution requiring explicit finite authority. There is no source-clip/golden-prefix dependency; the portable tensor-interface module and HTTP UI no longer import the legacy phrase worker/assets.

Core V2 uses `PersistentEpisode(backend,session_id,max_control_steps=...,cancel_event=...)` and its own `PreviewSession`; no V1/V2 episode mixing. The HTTP setup page does not start execution workers. Runtime/backend physical behavior is not validated by pure fixtures or AST-equivalence checks. A separately recorded native short HOLD/parity gate has since passed; new CLI/live-task/reach validation remains separately required. Record failures and do not switch back to private initialization silently.

## Explicit first-use CLI (new, execution not automatic)

`run_body_agent_v5.py` now composes the portable hybrid backend and coreV2. A focused command-line workflow is supplied before pretending the setup page is a completed studio. The page is a setup/admission dashboard with no active generation buttons or blank robot stage. The separately recorded agent demo remains separate.

The root's earlier portable short native gate passed; it is evidence only for that exact finite gate, not this new CLI, a fresh virtual environment, general reach or dance. This CLI requires its own reviewed plan digest. It does not read private Wave/High5/golden caches and has no private Codex CLI dependency.

HOLD request (engineering only; no language model):

```json
{"schema":"pulsemorph.portable_firstuse_request.v5","mode":"hold","instruction":"Maintain the frozen wrist position for one short sampled hold.","task":null,"provider":null}
```

```powershell
& .venv/Scripts/python.exe -I -B experiments/agentic/embodied/run_body_agent_v5.py --config body-config.json --request hold-request.json --output ./local/body-jobs-v5/hold_001 --plan hold-plan.json
# Inspect exact source/profile/native pins and budgets. Only after explicit review:
& .venv/Scripts/python.exe -I -B experiments/agentic/embodied/run_body_agent_v5.py --execute --plan hold-plan.json --approved-plan-sha256 <reviewed-sha256>
```

HOLD has two1second constant-reference prelude actions then a.4second measured hold: at most120online+1warmup,2,400actualsteps,121runtimeFK,242wristJacobian reads (failure terminal allowance1FK/2Jacobian),0model/sandbox/render calls. The target is explicitly frozen from the actual wrist after prelude, before the measured hold, with.012m tolerance and.2s sampled-hold criterion. It is not a user dance task or fabricated live model action.

Task mode accepts an exact `tasks_v2` wrist-task DTO under `task`, fixed before model calls, plus bounded instruction and explicit `provider`=`openrouter_free` or `nvidia_nim`. Environment keys only: `OPENROUTER_API_KEY` or `NVIDIA_API_KEY`; no configuration file contains them. The adapters use their actual model identities. No aliases pretend Nemotron is GPT, no paid fallback exists for the free selection, and no retries are made. At most4modelcalls/2previews,200actualcontrols including prelude/100sandboxcontrols,300online+1warmup,6,000totalsteps; the task verifier, parent-state binding and exact preview-commit checks remain coreV2's unchanged inner state machine. Input goals never become model-editable.

All output paths must be exclusive children of configured artifact_root. Durable attempt, budget, plan, actual trace, model response/transport receipts and final result are separate. Any failure stops; a failed output cannot be resumed or overwritten. Saved numeric checkpoint precedes trace export. Actual/sandbox inference budget is shared across clones, never reset per preview. Simulation pauses for model latency. No media rendering or aesthetic/dance success is implied by a numerical wrist task.

Fresh environment validation must separately confirm Python/wheel lock completeness, two unrelated workspace paths, no private histories/cache, source/native digest admission, no model on startup, unavailable-profile HTTP readiness, Host/Origin/token boundaries and then the finite native gate. Existing local recordings do not satisfy this new proof.
