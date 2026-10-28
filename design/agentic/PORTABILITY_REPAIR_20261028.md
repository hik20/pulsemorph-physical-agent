# Prepared-host portability repair: prospective V5

Read-only audit of public repository revision `3ca781ad42a52839c4f0f9d0a8784e8ed889cec4` and the corresponding local source. No runtime, asset, vendor, learning, history or public repository was changed. No install, download, model inference, FK or physics was run. Paths below are repository-relative or configuration names, not this host's private absolute paths.

## Concrete current failures

1. `experiments/agentic/embodied/start_body_agent_v4.ps1` ignores a newly created `.venv`: it selects a prepared cache interpreter. `embodied/app/manager_v1.py` then explicitly rejects every alternate interpreter, even one with identical dependency versions.
2. That launcher calls `app/phrase/entrypoint_v2.prepare_parent_imports` before serving UI. The helper constructs `PhraseAssets`, so even a read-only UI cannot start without the admitted private phrase cache.
3. `app/phrase/assets.py` fixes a developer-specific absolute cache path, interpreter path and manifest digest. Verification requires Wave, High5 and golden-prefix packages unrelated to the new body task. `embodied/backend_v1.py` actually consumes only `initial_carrier` to initialize the body; it does not play either source choreography.
4. `prepare_phrase_assets.py` builds the cache from historical local job outputs. It is an ordinary-copy recipe, not public acquisition. Its manifest includes historical absolute paths. Publishing it would neither make those paths available nor establish motion rights.
5. Public `docs/DEPENDENCIES.md` points at `asset_identity.json`, but that file is absent. The direct-version list is observed versions, not a platform wheel lock or a complete fresh-environment proof.

Do not fix these by disabling hashes, accepting any interpreter, copying private histories into the repository, or claiming the existing private cache is reproducible from a fresh checkout.

## Smallest additive implementation

Introduce versioned V5 launcher/config/profile/manager/backend entrypoints; retain all V1–V4 bytes and recorded authorities. A new backend adapter is necessary because backend V1 imports the phrase assets directly; changing only the launcher cannot repair the dependency chain.

Configuration DTO: exact fields for `runtime_python`, `vendor_root`, `profile_dir`, `artifact_root`, loopback `port`, and expected immutable public `asset_identity` digest. Resolve relative paths against the config file. Reject links, path escapes, unknown fields and conflicting CLI/config values. No secret fields. Keys stay in process environment and are not serialized.

V5 startup should serve read-only UI with `ready:false` and a precise unavailable category when prerequisites are absent. An explicit metadata preflight admits execution only after verifying configured interpreter identity, CPython version, isolation/user-site state, dependency versions and native-module hashes, source bytes, upstream commit and every asset digest. Create per-install environment receipts bound into each job's authority; a different machine's binary hash is expected to differ and must require explicit new admission, not pretend to reproduce this host's binary receipt. Do not auto-install or auto-run simulation on startup.

Separate public **body initialization profile** from phrase references. Proposed contents: a source-independent initial qpos36/qvel35 plus named reference joint q29/dq29 and required orientation carriers, generated from an explicitly specified owned standing pose; exact array names/dtypes/shapes/frame semantics; body/controller/model identities; generation script identity; no historical paths. A new deterministic standing pose must pass finite/range/frame checks and a separately authorized positive HOLD/reach pilot before it replaces Wave-frame0. It cannot inherit old byte-parity claims. An alternative public Wave-frame0 recipe retains that clip's unresolved data rights and is not the preferred rights-independent solution.

## Public asset identity draft

Proposed schema `pulsemorph.body_asset_identity.v1`: `upstream:{url,commit}`, `files:[{path,bytes,sha256,source_blob_sha256,materialization,license_pointer}]`, `initialization:{recipe_sha256,arrays_sha256,frame_layout,qualification}`, `environment_contract`, `scope`. Use relative paths only; a canonical manifest digest is explicitly admitted by V5. File digests prove identity, not rights or suitability.

Upstream source: [NVlabs/ProtoMotions](https://github.com/NVlabs/ProtoMotions), commit `f69fdcaabff31846b4d2247207417b57dc788828`. Existing checkout proves the following tracked artifacts; fresh acquisition has not been performed in this audit.

| Relative upstream asset | Required current raw SHA256 | Origin / materialization |
|---|---|---|
| `data/pretrained_models/motion_tracker/g1-bones-deploy/compiled_models/unified_pipeline.onnx` | `a59baa3e04a951e5cf0b4cc68f24ebaafa9272714226618b99a5017dfc805b4c` | Shipped compiled model,22,606,590B; use this export, not a re-export assumed identical |
| Same directory `unified_pipeline.yaml` | `57d81b60dda9aeb30c57d3428a822f203a83bca399c5abe82ffea7a9aaf8b139` | Tracked LF blob SHA `9b7896f3355a9d9d5e7d3139b83924eeb2e45c62c30bfda44afe996cfc6cf01c`; deterministic LF→CRLF gives required6,055B |
| `protomotions/data/assets/mjcf/g1_holo_compat.xml` | `e3926a6578cafaee38d0a7a3b7e7ebb0031f623c69e5cd34aa97d61e943b7e8e` | Tracked LF blob SHA `f9a455ff0b2822b3d3f68e55b02cd7e21c601a84cf3173237b7bfaf494492846`; deterministic LF→CRLF gives required41,005B |
| `protomotions/data/assets/mesh/G1/*.stl` | All63 exact file digests required | Current closure can be enumerated from pinned Git files; include individual relative path/bytes/SHA in the eventual JSON, not a wildcard-only identity |
| `deployment/{test_tracker_mujoco,tracker_inputs,state_utils,motion_utils}.py` | Individual pinned bytes required | Tracked deployment helpers; declare LF/CRLF materialization per file before hashing |

The exact XML and YAML already exist in the pinned official tree; no private XML generator or closed DanceModel implementation is needed. Model export documentation names `deployment/export_bm_tracker_onnx.py`, but regeneration is not proved byte-equivalent to the shipped export. The large `last.ckpt` is unnecessary for ONNX inference; the current local file is only an LFS pointer, not a loaded checkpoint.

Observed licenses: ProtoMotions source `LICENSE.md` Apache-2.0; model directory `MODEL_CARD.md`, `LICENSE.pdf`, `NOTICE.txt` specify NVIDIA Open Model License for weights/configuration/exports; Unitree G1 `mesh/G1/LICENSE` is BSD-3-Clause and upstream third-party notices cover G1 meshes and `g1_*.xml`. Preserve notices and review distribution conditions separately. Wave/High5 are shipped Kimodo examples with no established per-clip generation/license chain; do not bundle them or derived private phrase cache by default. This is provenance, not legal clearance.

## Prospective acquisition and validation sequence

These commands are a proposal, not commands run or a tested clean installation:

```powershell
py -3.11 -m venv .venv
# Install an approved platform-specific lock; current requirements-observed.txt is not that lock.
git clone --no-checkout https://github.com/NVlabs/ProtoMotions research/vendor/ProtoMotions
git -C research/vendor/ProtoMotions config core.autocrlf false
git -C research/vendor/ProtoMotions checkout f69fdcaabff31846b4d2247207417b57dc788828
# Future owned materializer: verify tracked LF digests, produce only declared CRLF assets,
# resolve LFS only for explicitly required files, then verify all63 mesh/model/helper hashes.
# Future V5 metadata-only preflight --config local-config.json --json
# Future V5 source-only/mock tests under .venv Python -I -B
# Separately approved finite native HOLD, then preview/commit parity, then one bounded reach.
```

Use `GIT_LFS_SKIP_SMUDGE=1` for prospective acquisition until exact required LFS paths and download sizes are reviewed; never `git lfs pull` the whole repository by default. Determine whether each required blob is ordinary Git or LFS before downloading. Keep the learned system, model weights, vendor tree and frozen body profile read-only. Do not publish host-specific configuration or original private initialization NPZ.

Validation must include two unrelated temporary workspace paths, a configured interpreter outside the old cache, absent-profile UI readiness, wrong source/model/native hashes rejected before controller construction, unknown config/linked paths rejected, and no silent fallback. A successful new HOLD establishes only its bounded physical scope. Old recordings remain evidence for the old prepared host, not the portable V5.

Next concrete work: generate the missing sanitized asset-identity JSON with full63-mesh/helper inventory; add V5 config + body-only profile recipe; obtain owner review of acquisition/rights; create a locked fresh environment; run metadata/mocks; only then seek the finite native admission gate. No publication is performed by this audit.
