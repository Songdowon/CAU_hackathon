import importlib
from pathlib import Path
import unittest
import torch
import yaml

class S06Tests(unittest.TestCase):
 def module(self):
  try: return importlib.import_module('S06')
  except ModuleNotFoundError: self.fail('S06 CKA Target-Floor is not implemented')
 def test_below_floor_has_zero_value_and_gradient(self):
  s=self.module(); x=torch.tensor(.019,requires_grad=True); y=s.cka_target_floor(x,.02); y.backward()
  self.assertEqual(y.item(),0); self.assertEqual(x.grad.item(),0)
 def test_above_floor_keeps_original_gradient(self):
  s=self.module(); x=torch.tensor(.037,requires_grad=True); y=s.cka_target_floor(x,.02); y.backward()
  self.assertAlmostEqual(y.item(),.017,places=6); self.assertEqual(x.grad.item(),1)
 def test_zero_floor_is_baseline_equivalent(self):
  s=self.module(); x=torch.tensor(.037,requires_grad=True); a=s.cka_target_floor(x,0); ga=torch.autograd.grad(a,x)[0]
  self.assertEqual(a.item(),x.item()); self.assertEqual(ga.item(),1)
 def test_boundary_uses_zero_subgradient(self):
  s=self.module(); x=torch.tensor(.02,requires_grad=True); s.cka_target_floor(x,.02).backward()
  self.assertEqual(x.grad.item(),0)
 def test_invalid_floor_or_cka_is_rejected(self):
  s=self.module()
  for floor in [-.1,1.1,float('nan'),float('inf')]:
   with self.assertRaises(ValueError): s.cka_target_floor(torch.tensor(.2),floor)
  for value in [float('nan'),float('inf')]:
   with self.assertRaises(ValueError): s.cka_target_floor(torch.tensor(value),.02)
 def test_config_matches_r017_except_floor_and_outputs(self):
  base=yaml.safe_load(Path('configs/r017.yaml').read_text()); cfg=yaml.safe_load(Path('configs/S06.yaml').read_text())
  self.assertEqual(cfg['script'],'S06.py'); self.assertEqual(cfg['train'].pop('cka_floor'),.02)
  cfg.pop('script'); cfg.pop('output'); base.pop('output'); self.assertEqual(cfg,base)

if __name__=='__main__': unittest.main()
