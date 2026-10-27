"""Repair protocol barriers exercised with file receipts; no simulation calls."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/agentic'))
import repair_driver as driver
import repair_model_io as model_io
from evaluate_repair import selection_key
from execute_repair import validate_repair_proposal


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


class RepairDriverTests(unittest.TestCase):
    def test_model_prompt_request_is_utf8_json_without_model_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);identity=driver.identifier('wave_a','llm_feedback',0)
            obs=out/'observations'/(identity+'.json')
            write(obs,{'instruction':'Müziğe göre dört koşulda hareket üret.'})
            with patch.object(model_io,'observe',return_value=obs) as observe:
                request=model_io.prepare(out,'wave_a','llm_feedback',0)
            observe.assert_called_once_with(out,'wave_a','llm_feedback',0)
            persisted=json.loads((out/'model_requests'/(identity+'.json')).read_text(encoding='utf8'))
            self.assertEqual(persisted,request)
            self.assertIn('NEW UTF-8 JSON response',request['prompt'])
            self.assertIn(obs.resolve().as_posix(),request['prompt'])
            self.assertEqual(request['observation'],driver.link(obs))
            self.assertEqual((request['model'],request['fork_turns']),('gpt-6.1-sol','none'))
            self.assertFalse((out/'responses').exists())
            self.assertFalse((out/'model_calls').exists())
            self.assertFalse((out/'trials').exists())

    def test_four_actual_gate_proposal_copies_pass_repair_preflight(self):
        panel=ROOT/'artifacts/runs/agentic/robustness_initial_state_v2'
        protocol=driver.read(panel/'protocol.json')
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);write(out/'protocol.json',{'synthetic_preflight_only':True})
            for role in ('calibration_wave','calibration_high5','selected_wave_a','selected_high5_b'):
                old=panel/'trials'/('nominal__'+role)/'proposal.json'
                raw=driver.read(old)
                seed=dict(path=old.relative_to(ROOT).as_posix(),sha256=driver.sha256(old))
                candidate=dict(candidate_id='gate_'+role,task_id=raw['task_id'],method='surrogate_search',
                               control_tool=raw['control_tool'],parameters=[raw['parameters'][name] for name in protocol['parameter_names']])
                scenario=next(s for s in protocol['scenarios'] if s['scenario_id']=='nominal')
                proposal_path=driver.trial_proposal(out,protocol,candidate,scenario,'gate__'+role,seed)
                proposal=driver.read(proposal_path)
                origin,content=validate_repair_proposal(proposal)
                self.assertEqual(origin,seed)
                self.assertEqual(content,old.read_bytes())
                self.assertEqual(proposal['parameters'],raw['parameters'])
                self.assertEqual(proposal['source_motion'],raw['source_motion'])
                self.assertEqual(proposal['task_id'],raw['task_id'])

    def test_observation_gate_barrier_before_any_task_access(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(driver,'context',return_value={}):
            out=Path(temp)
            with self.assertRaises(FileNotFoundError):
                driver.observation_data(out,'wave_a','llm_feedback',0)
            write(out/'gate.json',{'passed':False})
            with self.assertRaisesRegex(ValueError,'gate'):
                driver.observation_data(out,'wave_a','llm_feedback',0)

    def test_all_sixteen_off_commits_are_required_and_hash_checked(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)
            with self.assertRaises(FileNotFoundError):driver.precommit_off(out)
            for task in driver.TASKS:
                for index in range(8):
                    identity=driver.identifier(task,'llm_feedback_off',index)
                    obs=out/'observations'/(identity+'.json'); response=out/'responses'/(identity+'.json')
                    write(obs,{'observation':identity});write(response,{'response':identity})
                    proposal=out/'proposals'/(identity+'.json')
                    links=dict(observation=driver.link(obs),response=driver.link(response))
                    write(proposal,links)
                    write(out/'events'/('proposal_committed__'+identity+'.json'),dict(proposal=driver.link(proposal)))
                    write(out/'model_calls'/(identity+'.json'),dict(candidate_id=identity,model='gpt-6.1-sol',fork_turns='none',**links))
            self.assertTrue(driver.precommit_off(out))
            receipt_path=out/'model_calls'/(identity+'.json')
            receipt=driver.read(receipt_path)
            write(receipt_path,{**receipt,'candidate_id':'wrong'})
            with self.assertRaisesRegex(ValueError,'OFF call receipt changed'):driver.precommit_off(out)
            write(receipt_path,receipt)
            proposal_contents=driver.read(proposal)
            write(proposal,{**proposal_contents,'uncommitted':'tamper'})
            with self.assertRaisesRegex(ValueError,'OFF proposal commitment changed'):driver.precommit_off(out)
            write(proposal,proposal_contents)
            self.assertTrue(driver.precommit_off(out))
            write(response,{'response':'tampered'})
            with self.assertRaisesRegex(ValueError,'OFF commitment changed'):driver.precommit_off(out)

    def commit_fixture(self,out):
        identity=driver.identifier('wave_a','llm_feedback_off',0)
        obs=out/'observations'/(identity+'.json');response=out/'responses'/(identity+'.json')
        write(out/'protocol.json',{})
        write(obs,{'frozen':'observation'})
        write(response,dict(control_tool='global_clock_v1',parameters=[1,1,1,1,0],
                            only_observation_path=str(obs.resolve()),brief_reason='fixture'))
        receipt=dict(model='gpt-6.1-sol',fork_turns='none',candidate_id=identity,
                     observation=driver.link(obs),response=driver.link(response))
        write(out/'model_calls'/(identity+'.json'),receipt)
        p=dict(control_tools=['global_clock_v1'],parameter_names=['left_gain_a','right_gain_a','left_gain_b','right_gain_b','timing_shift_s'])
        return identity,obs,response,receipt,p

    def test_receipt_matches_fresh_response_and_off_precedes_physics(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);identity,obs,response,receipt,p=self.commit_fixture(out)
            with patch.object(driver,'context',return_value=p),patch.object(driver,'observation_data',return_value={'frozen':'observation'}):
                write(response,{**driver.read(response),'brief_reason':'changed after model receipt'})
                with self.assertRaisesRegex(ValueError,'artifact receipt'):
                    driver.commit(out,'wave_a','llm_feedback_off',0)
                receipt['response']=driver.link(response)
                write(out/'model_calls'/(identity+'.json'),receipt)
                write(out/'events/trial_start__candidate__already.json',{})
                with self.assertRaisesRegex(ValueError,'OFF must precede'):
                    driver.commit(out,'wave_a','llm_feedback_off',0)
                (out/'events/trial_start__candidate__already.json').unlink()
                row=driver.commit(out,'wave_a','llm_feedback_off',0)
                self.assertEqual(row['response'],driver.link(response))

    def test_confirmation_needs_full_frozen_selection_before_run(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(driver,'context',return_value={}),patch.object(driver,'run_trial') as physics:
            out=Path(temp);write(out/'selection.json',dict(winners=[{}]*5,confirmation_used_for_selection=False))
            with self.assertRaisesRegex(ValueError,'all6'):
                driver.confirm(out)
            physics.assert_not_called()
            write(out/'selection.json',dict(winners=[dict(task_id='wave_a',method='surrogate_search')]*6,confirmation_used_for_selection=False))
            with self.assertRaisesRegex(ValueError,'distinct'):
                driver.confirm(out)
            physics.assert_not_called()
            write(out/'protocol.json',{})
            selected=dict(winners=[dict(task_id=task,method=method) for task in driver.TASKS for method in driver.METHODS],
                          confirmation_used_for_selection=False,protocol_sha256=driver.sha256(out/'protocol.json'))
            write(out/'selection.json',selected)
            with self.assertRaises(FileNotFoundError):driver.confirm(out)
            physics.assert_not_called()
            write(out/'events/selection_frozen__all6.json',dict(event='selection_frozen',selection=dict(path='wrong',sha256='a'*64)))
            with self.assertRaisesRegex(ValueError,'freeze receipt'):driver.confirm(out)
            physics.assert_not_called()

    def test_measured_missing_accent_ranks_without_technical_abort(self):
        missing=dict(integrity_passed=True,physical_gate_passed=True,utility=None,utility_l2=None)
        available={**missing,'utility':.8,'utility_l2':1.2}
        self.assertGreater(selection_key(missing,'a'),selection_key(available,'b'))
        failed={**available,'physical_gate_passed':False,'utility':0}
        self.assertLess(selection_key(missing,'a'),selection_key(failed,'b'))

    def test_planned_unique_trial_ids_are_250(self):
        old=driver.read(ROOT/'artifacts/runs/agentic/robustness_initial_state_v2/protocol.json')
        conditions=[s['scenario_id'] for s in old['scenarios']]
        confirmation=[s for s in conditions if s not in driver.DEV]
        ids={'gate__'+role for role in ('calibration_wave','calibration_high5','selected_wave_a','selected_high5_b')}
        ids.update('candidate__'+driver.identifier(task,method,index)+'__'+condition
                   for task in driver.TASKS for method in driver.METHODS for index in range(8) for condition in driver.DEV)
        ids.update('confirmation__'+task+'__'+method+'__'+condition
                   for task in driver.TASKS for method in driver.METHODS for condition in confirmation)
        self.assertEqual(len(ids),250)
        self.assertEqual(len(driver.DEV),4);self.assertEqual(len(confirmation),9)


if __name__=='__main__':unittest.main()
