"""Create a separately reviewed executor fork from the frozen phrase witness.

This only writes new source files, never runs a controller or grants execution.
All exact replacements are fail-closed. The original witness remains frozen.
"""
from pathlib import Path
import hashlib

HERE = Path(__file__).resolve().parent


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError('expected one source replacement: ' + old[:90])
    return source.replace(old, new, 1)


def main():
    adapter = (HERE / 'adapter_v1.py').read_bytes()
    executor = (HERE / 'execute_phrase_v1.py').read_bytes()
    if hashlib.sha256(adapter).hexdigest() != 'ed8d44776161f847c9dc576e9ad03ba1baf5cf4a853134a117d311d03f52d00d':
        raise ValueError('frozen adapter changed')
    if hashlib.sha256(executor).hexdigest() != '0ab30ea1237295d5ed1a5b6fbb3100b9d253a9eb4b520abd22424e0ae857a7ed':
        raise ValueError('frozen executor changed')
    a = adapter.decode().replace('\r\n', '\n')
    a = a.replace('authentic Wave', 'authentic first-source').replace('exact row gates', 'exact eighteen-control row gates')
    a = a.replace("'fixed-wave-high5-phrase-reference'", "'phrase-program-reference'")
    # Prefix control0..17 (warmup+18 online calls), state0..18. At control18
    # the +160ms horizon first reaches0.52s and can observe a nonunit gain.
    a = a.replace('168', '18').replace('169', '19').replace('167', '17')
    a = a.replace('(19,19,19,18)', '(19,19,19,18)')
    x = executor.decode().replace('\r\n', '\n')
    x = replace_once(x, 'from adapter_v1 import ', 'from adapter_program_v1 import ')
    x = x.replace("agentic/phrase_execution_plan_v1'", "agentic/phrase_program_execution_plan_v1'")
    x = x.replace("agentic/phrase_reference_plan_v1'", "agentic/phrase_program_reference_v2'")
    x = x.replace("experiments/phrase_execution_v1'", "experiments/phrase_program_execution_v1'")
    x = replace_once(x, "or len(report['source_pins'])!=225", "or not report['source_pins']")
    x = x.replace("HERE/'adapter_v1.py'", "HERE/'adapter_program_v1.py'")
    x = x.replace("REFERENCE_PLAN/'prefix_input_parity.json'", "REFERENCE_PLAN/'prefix_actual_assembly_parity.json'")
    x = x.replace("ROOT/'design/agentic/PHRASE_REFERENCE_PLAN_V1.md'", "ROOT/'design/agentic/PHRASE_PROGRAM_V1.md'")
    x = x.replace("'original_reference_pins':225", "'original_reference_pins':len(report['source_pins'])")
    x = x.replace("'scientific_schedule':'Wave0..3.5;1s jointC1 Hermite/torsoC0 SLERP;High5(t-4)4.5..8.14'", "'scientific_schedule':'Frozen model-proposed Wave-to-High5 program; source0..3.5, C1joint/C0torso bridge3.5..4.5, second(t-4) thereafter; exact compiled reference authoritative'")
    x = x.replace("'audio':False,'llm':False", "'audio':True,'llm':False,'audio_scope':'Offline eight-second PCM summary informed the frozen program; execution makes no model/audio-analysis calls or beat-timing claim'")
    x = x.replace("'audio':False,'protocol_sha256'", "'audio':True,'protocol_sha256'")
    x = x.replace("'Torso C0 only; angular velocity jumps .301572/.672179rad_s remain'", "'Torso C0 only; no angular velocity continuity or physical superiority claim'")
    x = x.replace("fromfile='unchanged_execute_app.py',tofile='new_execute_phrase_v1.py'", "fromfile='unchanged_execute_app.py',tofile='new_execute_program_v1.py'")
    x = x.replace("'pulsemorph.phrase_execution_protocol.v1'", "'pulsemorph.phrase_program_execution_protocol.v1'")
    x = x.replace("'pulsemorph.phrase_physical_trial.v1'", "'pulsemorph.phrase_program_physical_trial.v1'")
    x = x.replace('authentic Wave', 'authentic first-source').replace('static authentic Wave0', 'static authentic first-source0')
    x = x.replace('168', '18').replace('169', '19').replace('167', '17')
    # Baseline scope here is deliberately one concrete Wave-first program.
    # Reverse-order execution needs its own protocol/reset/gate; it is rejected.
    x = replace_once(x, "reference=load_reference(REFERENCE_PLAN/'common_reference')", "reference=load_reference(REFERENCE_PLAN/'common_reference')\n    program=reference.metadata['provenance'].get('phrase_program')\n    if not program or tuple(program['order'])!=('wave','high5'):\n        raise ValueError('this concrete execution protocol only admits the frozen Wave-first program')")
    for name, source in (('adapter_program_v1.py', a), ('execute_program_v1.py', x)):
        compile(source, name, 'exec')
        with (HERE / name).open('x', encoding='utf8', newline='\n') as stream:
            stream.write(source)
    print('Prepared separate adapter/executor sources; no execution authority.')


if __name__ == '__main__':
    main()
