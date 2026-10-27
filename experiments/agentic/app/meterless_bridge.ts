/** JSON stdin/stdout bridge to the pinned, unmodified local Meterless engine. */
import fs from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { WorldModel } from "../../../research/vendor/Meterless/engines/world-model/reference/src/index.ts";

async function main() {
const root = path.resolve(__dirname, "../../..");
const vendor = path.join(root, "research/vendor/Meterless");
const commit = "1742dec56a66e58eef9dda2d2b865823758ca8cd";
if (execFileSync("git", ["-C", vendor, "rev-parse", "HEAD"], { encoding: "utf8" }).trim() !== commit)
  throw new Error("Meterless source version mismatch");
execFileSync("git", ["-C", vendor, "diff", "--exit-code", "HEAD", "--", "engines/world-model/reference/src"]);
const request = JSON.parse(fs.readFileSync(0, "utf8"));
const directory = path.resolve(request.directory);
const allowed = path.join(root, "artifacts/runs/agentic");
if (!directory.startsWith(allowed + path.sep)) throw new Error("App memory must stay under agentic artifacts");
const options = { namespace: "pulsemorph-agentic-app-v1", storage: "local" as const, directory };
const world = new WorldModel(options);
if (request.operation === "record") {
  const record = request.record;
  if (!["app_job", "app_trial"].includes(record.type) || !record.id) throw new Error("Invalid app memory record");
  await world.upsertEntity({ type: record.type, externalKey: { system: "pulsemorph-app", id: record.id },
    name: record.id, attrs: record, source: { kind: "application", by: "pulsemorph-agentic-app", at: record.updated_at, runId: record.job_id ?? record.id } });
} else if (request.operation !== "list" && request.operation !== "verify") throw new Error("Unknown operation");
const rows = world.view("graph").query({ where: e => e.type === "app_job" || e.type === "app_trial" });
const reopened = new WorldModel(options);
if (world.stateShapeCanonical() !== reopened.stateShapeCanonical()) throw new Error("Memory reopen mismatch");
console.log(JSON.stringify({ provider: "meterless", mode: "local_upstream", commit, persisted_reopen_equal: true,
  event_count: world.events().length, records: rows.map(e => e.attrs) }));
}
main().catch(() => { console.error("Meterless app bridge failed"); process.exitCode = 1; });
