"""Read-only local CSV/schema/evidence check; no network, account or model/physics."""
import csv,hashlib,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    directory=Path(__file__).parent/'registry_v1';manifest=json.loads((directory/'manifest.json').read_text())
    for name,digest in manifest['outputs_sha256'].items():assert sha(directory/name)==digest,name
    with (directory/'capabilities.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    import jsonschema
    schema=json.loads((directory/'schema.json').read_text())
    assert len(rows)==2 and {r['source_id'] for r in rows}=={'wave','high5'}
    for row in rows:jsonschema.validate(row,schema);assert json.loads(row['capability'])['body_profile_id']==row['body_id']=='g1'
    assert sha(Path(__file__).parent/'build_capability_registry.py')==manifest['builder_sha256']
    for name,digest in manifest['input_evidence_sha256'].items():assert sha(ROOT/name)==digest,name
    print(json.dumps({'passed':True,'records':len(rows),'csv_roundtrip':True,'evidence_hashes_verified':len(manifest['input_evidence_sha256']),'remote_queries':0,'model_calls':0,'physics_steps':0}))
if __name__=='__main__':main()
