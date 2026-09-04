import contextlib
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import yaml
from tools import S04_run as runner


class RunnerTests(unittest.TestCase):
    def test_complete_suite_uses_paired_settings_and_local_reports(self):
        original=Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ('S04-control','S04'):
                cfg=yaml.safe_load((original/'configs'/(name+'.yaml')).read_text())
                (root/'configs').mkdir(exist_ok=True)
                (root/'configs'/(name+'.yaml')).write_text(yaml.safe_dump(cfg))
            tracked=['S04.py','S04_reference.py','tools/S04_run.py','tools/run_exp.py','tools/fasteval.py','score_model.py']
            for path in tracked:
                p=root/path; p.parent.mkdir(parents=True,exist_ok=True); p.write_text('# test fixture')
            order=[]
            fast={'Acc_f':.1,'Acc_r':95.6,'CKA_f_o':.02,'CKA_r_o':.96,'AUS':.99,'RUS_o':.97,'final':.98}
            diagnostics={'steps':4800,'conflicts':100,'projections':0,'actual_update_samples':49,'positive_retain_linear_changes':0}
            def train(config_path):
                name=Path(config_path).stem; order.append(name)
                cfg=yaml.safe_load(Path(config_path).read_text())
                d={**diagnostics,'projections':100 if name=='S04' else 0}
                runner.write_json(cfg['output']['report'],{'status':'training_completed_evaluation_pending','diagnostics':d,'elapsed_seconds':1})
                return {**fast,'final':.981 if name=='S04' else .98}
            def gpu(command,on_ready):
                on_ready()
                if '--smoke' in command:
                    order.append('smoke')
                    runner.write_json('results/S04-smoke.run.json',{'status':'smoke_completed','diagnostics':{'steps':4},'config':{'data':{'batch_size':128}},'cuda_projection_sanity':{'passed':True}})
                    return subprocess.CompletedProcess(command,0,stdout='',stderr='')
                name=Path(command[-1]).stem; order.append('full:'+name)
                payload={'phase':'validation','tag':name+'.pt','AUS':.99,'RUS_o':.97,'final_score':.981 if name=='S04' else .98,
                         'accuracy_metric':{'Acc_f':.001,'Acc_r':.956},'representation_metric':{'CKA_f_o':.02,'CKA_r_o':.961 if name=='S04' else .96}}
                report='results/'+name+'-validation-test.json'; runner.write_json(report,payload)
                return subprocess.CompletedProcess(command,0,stdout='report      : '+report,stderr='')
            try:
                with patch.object(runner,'ROOT',root),patch.object(runner.run_exp,'run',side_effect=train),patch.object(runner,'gpu_command',side_effect=gpu),patch.object(sys,'argv',['S04_run.py']),contextlib.redirect_stdout(io.StringIO()):
                    runner.main()
                state=json.loads((root/'logs/S04.state.json').read_text())
                result=json.loads((root/'results/S04.comparison.json').read_text())
                self.assertEqual(state['status'],'completed')
                self.assertEqual(order,['smoke','S04-control','S04','full:S04-control','full:S04'])
                self.assertAlmostEqual(result['score_model_delta_vs_control']['final'],.001)
                self.assertAlmostEqual(result['score_model_delta_vs_control']['CKA_r_o'],.001)
                self.assertTrue((root/'results/S04.comparison.md').is_file())
                with patch.object(runner,'ROOT',root),patch.object(runner.run_exp,'run') as rerun,patch.object(sys,'argv',['S04_run.py']):
                    with self.assertRaises(FileExistsError): runner.main()
                    rerun.assert_not_called()
            finally: os.chdir(original)


if __name__=='__main__': unittest.main()
