"""Desired physical ratios are data, distinct from bounded reference controls."""
import math
SCHEMA='pulsemorph.phrase_goals.v1'
GOALS_SCHEMA={'type':'object','additionalProperties':False,'properties':{'schema':{'type':'string','enum':[SCHEMA]},'ratios':{'type':'array','minItems':4,'maxItems':4,'items':{'type':'number','minimum':.75,'maximum':1.25}}},'required':['schema','ratios']}
def validate_goals(value):
    if type(value) is not dict or set(value)!={'schema','ratios'} or value['schema']!=SCHEMA or type(value['ratios']) is not list or len(value['ratios'])!=4:raise ValueError('Exact phrase goals schema and four ratios required')
    if any(type(v) not in (int,float) or not .75<=v<=1.25 or not math.isfinite(v) for v in value['ratios']):raise ValueError('Goal ratios must be finite numbers in [.75,1.25], not bool')
    return {'schema':SCHEMA,'ratios':[float(v) for v in value['ratios']]}
