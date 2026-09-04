import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import yaml
from tools import S05_run as runner


class SuiteTests(unittest.TestCase):
    def exercise(self,feasible):
        original=Path.cwd(); cfg=yaml.safe_load((original/'configs/S05.yaml').read_text())
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            tracked=['S05.py','tools/S05_run.py','tools/run_exp.py','tools/fasteval.py','score_model.py','imagenet_vit.py','train_ft.py',
                     'splits/student_split.pt','validation_cache/refs.pt','validation_cache/M_o__validation.npz','cache/val_px_fp16.npy','cache/val_labels.npy']+[s['path'] for s in cfg['sources']]
            tracked+=['scorer/'+p for p in ('score_unlearning.py','convert_checkpoint.py','imagenet_vit.py')]
            for name in tracked:
                p=root/name; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(b'fixture')
            (root/'configs').mkdir(exist_ok=True); (root/'configs/S05.yaml').write_text(yaml.safe_dump(cfg))
            (root/'results').mkdir(exist_ok=True)
            def payload(name,aus=.996,rus=.98):
                return {'phase':'validation','tag':name,'AUS':aus,'RUS_o':rus,'final_score':2*aus*rus/(aus+rus),'dataset_revision':'fixture','score_version':'fixture',
                        'accuracy_metric':{'Acc_f':.2,'Acc_r':95.7},'representation_metric':{'CKA_f_o':.02,'CKA_r_o':.98}}
            for h in cfg['historical_reports']: (root/h['report']).write_text(json.dumps(payload(Path(h['checkpoint']).name)))
            rows={}; calls=[]
            def gpu(command,log_path,on_ready):
                on_ready(); calls.append(list(command))
                if '--candidate' in command:
                    ident=command[command.index('--candidate')+1]; start=command.index('--weights')+1; w=[float(v) for v in command[start:start+3]]
                    aus=.996 if feasible else .9945; rus=.99-.1*sum((a-b)**2 for a,b in zip(w,[.52,.27,.21]))
                    p=Path(cfg['output']['model_dir'])/('S05-'+ident+'.pt'); p.write_bytes(ident.encode())
                    record={'id':ident,'weights':w,'source_order':['r016','r015','r012'],'checkpoint':str(p),'checkpoint_sha256':runner.sha256(p),
                            'metrics':{'AUS':aus,'RUS_o':rus,'final':2*aus*rus/(aus+rus),'Acc_f':.2,'Acc_r':95.7,'CKA_f_o':.02,'CKA_r_o':.98},'seconds':.1}
                    runner.write_json(Path(cfg['output']['result_dir'])/('S05-'+ident+'.fast.json'),record,exclusive=True); rows[ident]=record
                    text=json.dumps(record)
                else:
                    ident=Path(command[-1]).stem.removeprefix('S05-'); row=rows[ident]; w=row['weights']
                    rejected=(abs(w[0]-.525)<1e-8 and abs(w[1]-.275)<1e-8)
                    aus=.9949 if rejected or not feasible else .995
                    path='results/S05-'+ident+'-validation-fixture.json'
                    runner.write_json(path,payload(Path(command[-1]).name,aus,row['metrics']['RUS_o']),exclusive=True)
                    text='report      : '+path
                with Path(log_path).open('x') as log: log.write(text)
                return text
            try:
                with patch.object(runner,'ROOT',root),patch.object(runner.run_exp,'LOG',root/'EXPERIMENTS.md'),patch.object(runner,'gpu_command',side_effect=gpu),patch.dict(os.environ,{'TRUSTED_SCORER_ROOT':str(root/'scorer')}),patch.object(sys,'argv',['S05_run.py']),contextlib.redirect_stdout(io.StringIO()):
                    runner.main()
                state=json.loads((root/cfg['output']['state']).read_text()); comp=json.loads((root/cfg['output']['comparison']).read_text())
                self.assertLessEqual(state['completed_fast'],27); self.assertGreaterEqual(state['completed_fast'],9)
                self.assertLessEqual(state['completed_full'],8); self.assertIn('C000',comp['shortlist'])
                self.assertEqual(len(rows),len({runner.weight_key(r['weights']) for r in rows.values()}))
                if feasible:
                    self.assertEqual(state['status'],'completed'); winner=comp['winner']
                    self.assertGreaterEqual(winner['metrics']['AUS'],.995)
                    eligible=[r for r in comp['candidates'] if r.get('full',{}).get('metrics',{}).get('AUS',0)>=.995]
                    self.assertEqual(winner['metrics']['RUS_o'],max(r['full']['metrics']['RUS_o'] for r in eligible))
                    self.assertEqual((root/cfg['output']['winner']).read_bytes(),winner['id'].encode())
                else:
                    self.assertEqual(state['status'],'completed_no_feasible_candidate'); self.assertIsNone(comp['winner']); self.assertFalse((root/cfg['output']['winner']).exists())
            finally: os.chdir(original)
    def test_confirmed_constraint_and_export(self): self.exercise(True)
    def test_no_feasible_candidate_does_not_export(self): self.exercise(False)


if __name__=='__main__': unittest.main()
