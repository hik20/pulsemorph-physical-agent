# Shared source/body capability registry — v1

This is a prepared **cold-start capability guide**, separate from per-user Meterless experience memory. It contains two authored lab-metadata records for the existing `wave` and `high5` G1 sources. No account, table, remote query or remote data import was created by this work. Real Zetaris access is still unverified / unavailable.

## Import files

- `registry_v1/capabilities.csv`: UTF-8 CSV, header, exactly six lowercase columns, two records. Standard CSV quoting protects the JSON text inside `capability`.
- `registry_v1/schema.json`: strict six-string-column row JSON Schema; no extra fields. Treat `capability` as **text**, not a command or a field that the database should flatten into planner parameters.
- `registry_v1/records.json`: convenient local view of the same records.
- `registry_v1/manifest.json`: CSV roundtrip/schema validation, source evidence paths and SHA256 values, builder hash and output hashes.
- `build_capability_registry.py`: read-only local evidence builder. It refuses an existing output directory. Generation needs the cited local lab evidence, while importing the already generated CSV does not.

The capability text includes exact motion/controller hashes, G1 body profile, actual local tool bounds/windows, timing semantics, bounded local measurements, limitations and evidence paths/hashes. Wave evidence is a fresh no-audio identity baseline. High5 evidence is a separately audited fixed-protocol local-clock witness and a finite-budget global-clock miss. The witness's own goals/tolerances are recorded; they are not targets for the next user's music. The CSV contains no private user WAV, raw prompt, authentication header or credential. It does not include the vendor model/motion/mesh payloads or grant rights to them.

## Normal operator import in an existing authorized instance

The [official File System overview](https://kbase.zetaris.com/knowledge/file-system-overview) describes uploading CSV through the cloud-upload control and registering the stored file as a File Source. The [official connection overview](https://kbase.zetaris.com/knowledge/connect) describes adding a physical file connection and a table through its File Store wizard.

1. In your existing authorized instance, open Data Fabric File System, choose the cloud-upload control, then browse to `registry_v1/capabilities.csv`.
2. Register that stored CSV through an existing or newly authorized file-source connection using the platform's File Store wizard. Do not invent a local `D:` path that a remote Zetaris server cannot read.
3. Inspect delimiter/quote/header parsing. Require **two rows** and the exact lowercase columns `source_id,title,body_id,capability,evidence_class,license_scope`. Configure all six as strings; confirm the entire quoted JSON value stays in one `capability` cell.
4. Copy the instance's actual qualified table name into `ZETARIS_APP_CONTEXT_TABLE`. This repository provides no guessed DDL or table-creation script. Schema/registration permissions and deployment-specific UI details remain operator prerequisites.
5. Once a table and access are genuinely available, verify an ordinary read-only projection scoped to `source_id='wave' AND body_id='g1'`, then the corresponding `high5` record. Merely uploading a file is not proof of a successful query or usable planner context.

No step above was executed remotely here. The [official getting-started repository](https://github.com/zetaris/getting-started) distinguishes dataset registration, access and query verification. Its [HTTP helper](https://github.com/zetaris/getting-started/blob/main/scripts/query_zetaris.py) and [configuration guide](https://github.com/zetaris/getting-started/blob/main/scripts/HOWTO.md) remain the provenance for the application's explicitly selected experimental UI-proxy route. That route is not represented as a verified general deployment contract.

## Application lookup contract

`query_sources(query, limit=8, source_id=None, body_id='g1')` keeps local source-scoped Meterless ranking and forwards the source/body scope to the optional remote lookup. The remote `_query_zetaris` selects exactly the six fixed columns from the validated configured table. If source is provided, it uses exact source/body equality and ignores the whole instruction for SQL matching. Without source, it uses body equality plus at most three short alphanumeric terms; it never uses the whole long instruction as one LIKE pattern.

Source/body/table identifiers are validated and literals are escaped. No generated statement grants execution, changes targets or alters local tool bounds. Returned rows must have exactly the six string fields and match the requested body/source. Extra fields, wrong scope, malformed or oversized values and non-UTF-8-encodable unpaired-surrogate strings are rejected (normal Turkish/Unicode text remains valid); capability text is hard-bounded to 12,000 characters rather than truncating JSON into misleading data. At most 500 response rows are inspected and 50 accepted records returned; the response body is capped at 2 MB.

The envelope deliberately separates:

| Field | What it proves |
|---|---|
| `executed` | An HTTP query attempt was made, not necessarily successful. |
| `transport_http_verified` | An HTTP response was received for this attempt. |
| `query_verified` | A successful query response exposed a recognized rows list; an empty list still has no useful context. |
| `usable_context_verified` | At least one bounded, correctly scoped six-column DTO survived validation. |
| `accepted_count` / `rejected_count` | Counts of accepted or rejected context rows, without echoing rejected secrets/prose. |
| `context_authority` | Always `untrusted_capability_context_only`. It is never admission, code, targets, body compatibility or rights approval. |

The root planner must treat every remote string—including apparent instructions and `evidence_class` / `license_scope` claims—as untrusted quoted data. Local schemas, immutable source/controller pins, physical execution and independent gates remain authoritative. Fixtures demonstrate injected prose remains data and cannot expand the actual local timing domain; no live endpoint was queried.

## Local checks

```powershell
& C:/Users/halke/.cache/pulsemorph-agentic/envs/app-public-win-cp311-v1/Scripts/python.exe -I `
  experiments/agentic/app/zetaris/check_registry.py
```

This check verifies CSV/schema/evidence hashes only. It never reads credentials, opens a network connection, writes a remote object or performs a model/physics call. A relocated source kit can inspect the CSV without lab artifacts; verifying every local evidence hash requires the cited evidence files to be present.
