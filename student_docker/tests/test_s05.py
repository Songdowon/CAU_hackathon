import importlib
import math
from pathlib import Path
import tempfile
import unittest
import torch


class SoupTests(unittest.TestCase):
    def module(self):
        try: return importlib.import_module('S05')
        except ModuleNotFoundError: self.fail('S05 Weighted Soup Search is not implemented')
    def test_weight_constraints(self):
        s=self.module()
        self.assertEqual(s.validate_weights([.5,.3,.2]),(.5,.3,.2))
        for w in ([.5,.3],[-.1,.5,.6],[.5,.3,.3],[float('nan'),.5,.5],[float('inf'),0,0],[True,0,0]):
            with self.assertRaises(ValueError): s.validate_weights(w)
    def test_analytic_average_endpoint_and_inputs_unchanged(self):
        s=self.module(); states=[{'w':torch.tensor([v,2*v]),'counter':torch.tensor(7)} for v in [1.,3.,8.]]
        before=[{k:v.clone() for k,v in d.items()} for d in states]
        out=s.weighted_state_dict(states,[.5,.3,.2])
        torch.testing.assert_close(out['w'],torch.tensor([3.,6.]))
        self.assertEqual(out['counter'].dtype,torch.int64)
        self.assertTrue(torch.equal(s.weighted_state_dict(states,[0,1,0])['w'],states[1]['w']))
        for a,b in zip(states,before):
            for k in a: self.assertTrue(torch.equal(a[k],b[k]))
    def test_identical_frozen_tensors_remain_exact(self):
        s=self.module(); torch.manual_seed(7)
        x=torch.randn(32,32); states=[{'w':x.clone()} for _ in range(3)]
        out=s.weighted_state_dict(states,[1/3,1/3,1/3])
        self.assertTrue(torch.equal(out['w'],x)); self.assertNotEqual(out['w'].data_ptr(),x.data_ptr())
    def test_state_mismatches_and_nonfinite_are_rejected(self):
        s=self.module(); valid={'w':torch.ones(2),'counter':torch.tensor(1)}
        for bad in ({'other':torch.ones(2)},dict(valid,w=torch.ones(3)),dict(valid,w=torch.ones(2,dtype=torch.float64)),dict(valid,counter=torch.tensor(2)),dict(valid,w=torch.tensor([float('nan'),1.]))):
            with self.assertRaises(ValueError): s.weighted_state_dict([valid,valid,bad],[.5,.3,.2])
    def test_refinement_preserves_simplex_and_deduplicates(self):
        s=self.module(); center=(.5,.3,.2)
        points=s.refinement([center],.05,seen=[center,(.55,.25,.2)])
        self.assertEqual(len(points),5)
        self.assertEqual(len(set(points)),5)
        for w in points: self.assertAlmostEqual(sum(w),1); self.assertTrue(all(v>=0 for v in w))
        self.assertEqual(len(s.refinement([(1.,0,0)],.05,seen=[])),2)
    def test_exact_final_constraint_and_rus_priority(self):
        s=self.module()
        rows=[{'id':'a','metrics':{'AUS':.994999,'RUS_o':.999,'final':.996}},
              {'id':'b','metrics':{'AUS':.995,'RUS_o':.983,'final':.989}},
              {'id':'c','metrics':{'AUS':.999,'RUS_o':.982,'final':.990}}]
        self.assertEqual(s.select_best(rows,.995)['id'],'b')
        self.assertIsNone(s.select_best(rows,1.0))
    def test_screening_allows_precision_margin_but_full_selection_does_not(self):
        s=self.module()
        rows=[{'id':'eq','metrics':{'AUS':.995,'RUS_o':.98,'final':.987}},
              {'id':'edge','metrics':{'AUS':.9945,'RUS_o':.99,'final':.992}},
              {'id':'far','metrics':{'AUS':.993,'RUS_o':.999,'final':.996}}]
        self.assertEqual(s.rank_screen(rows,.995,.001)[0]['id'],'edge')
        self.assertEqual(s.select_best(rows,.995)['id'],'eq')
    def test_checkpoint_write_preserves_existing_artifact(self):
        s=self.module()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'old.pt'; p.write_bytes(b'previous-result')
            with self.assertRaises(FileExistsError): s.save_checkpoint(p,{'w':torch.ones(2)})
            self.assertEqual(p.read_bytes(),b'previous-result')
    def test_nonfinite_metrics_rejected(self):
        s=self.module()
        with self.assertRaises(ValueError): s.select_best([{'id':'bad','metrics':{'AUS':.996,'RUS_o':float('nan'),'final':.99}}],.995)


if __name__=='__main__': unittest.main()
