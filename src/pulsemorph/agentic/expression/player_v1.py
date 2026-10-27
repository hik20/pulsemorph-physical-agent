"""Original expression producer/hash validation plus frozen player sampling."""
import hashlib,json
import numpy as np
from pulsemorph.motion.g1 import sample_reference
from pulsemorph.agentic.phrase_expression_v1 import parse_expression
from pulsemorph.agentic.phrase_reference import plain
from pulsemorph.agentic.app.phrase.worker import PhrasePlayer

def expression_hash(value):
    p=parse_expression(value)
    return hashlib.sha256(json.dumps(p,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf8')).hexdigest()
def validate_expression_reference(reference,expression):
    reference.validate();parsed=parse_expression(expression);digest=expression_hash(parsed);meta=reference.metadata
    if reference.time_offsets_s[-1]!=8.14 or meta['controller']['control_dt_s']!=.02 or meta['producer']['id']!='phrase-expression-reference' or meta['producer']['version']!='1':raise ValueError('Original expression1 producer/clock required')
    if parse_expression(plain(meta['provenance']['phrase_expression']))!=parsed or meta['provenance']['phrase_expression_sha256']!=digest or meta['context'].get('expression_sha256')!=digest:raise ValueError('Original expression request/reference hash binding mismatch')
    if parsed['order']!=['wave','high5']:raise ValueError('Expression physical admission only Wave→High5')
    return parsed

class ExpressionPlayer(PhrasePlayer):
    """Inherit exact old get_state/future methods; never call old producer shim."""
    def __init__(self,reference,initial_carrier,anchor_index,expression):
        validate_expression_reference(reference,expression)
        self.reference=reference
        self.initial={k:np.asarray(v).copy() for k,v in initial_carrier.items()}
        self.anchor=anchor_index
        if self.initial['body_pos'].shape!=(33,3) or self.initial['body_rot'].shape!=(33,4) or not all(np.isfinite(v).all() for v in self.initial.values()):raise ValueError('Actual Wave initial body carrier required')
        first=sample_reference(reference,np.array([0.]))
        if not np.array_equal(first['dof_pos'][0].astype('float32'),self.initial['dof_pos']) or not np.array_equal(first['anchor_rot'][0].astype('float32'),self.initial['body_rot'][anchor_index]):raise ValueError('Expression initial q/torso differs from authentic carrier')
