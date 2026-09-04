import copy
import importlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch import nn

class S04Tests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('S04'),'S04 gradient surgery is not implemented')
        self.api=importlib.import_module('S04')

    def test_conflict_removes_only_opposing_forget_component(self):
        r=[torch.tensor([1.,0.])]; f=[torch.tensor([-2.,3.])]
        merged,stats=self.api.combine_gradients(r,f,enabled=True,min_retain_norm=1e-8)
        torch.testing.assert_close(merged[0],torch.tensor([1.,3.]))
        self.assertTrue(stats['projected'])
        self.assertAlmostEqual(stats['dot_after'],0.,places=6)
        torch.testing.assert_close(r[0],torch.tensor([1.,0.]))
        torch.testing.assert_close(f[0],torch.tensor([-2.,3.]))

    def test_global_nonconflict_does_not_project_individual_blocks(self):
        merged,stats=self.api.combine_gradients([torch.tensor([1.]),torch.tensor([1.])],[torch.tensor([-2.]),torch.tensor([3.])],enabled=True,min_retain_norm=1e-8)
        torch.testing.assert_close(torch.cat(merged),torch.tensor([-1.,4.]))
        self.assertFalse(stats['projected'])

    def test_zero_and_tiny_retain_signal_skip_projection(self):
        for scale in [0.,1e-12]:
            merged,stats=self.api.combine_gradients([torch.tensor([scale])],[torch.tensor([-1.])],enabled=True,min_retain_norm=1e-8)
            self.assertTrue(torch.isfinite(merged[0]).all())
            self.assertFalse(stats['projected'])
            self.assertAlmostEqual(merged[0].item(),-1.,places=6)

    def test_unused_gradients_preserve_full_vector_projection(self):
        merged,stats=self.api.combine_gradients([torch.tensor([1.]),torch.tensor([2.]),None],[torch.tensor([-1.]),None,None],enabled=True,min_retain_norm=1e-8)
        torch.testing.assert_close(torch.cat(merged[:2]),torch.tensor([0.2,2.4]))
        self.assertIsNone(merged[2])
        self.assertAlmostEqual(stats['dot_after'],0.,places=6)

    def test_disabled_control_is_ordinary_gradient_sum(self):
        merged,stats=self.api.combine_gradients([torch.tensor([1.,2.])],[torch.tensor([-3.,-4.])],enabled=False,min_retain_norm=1e-8)
        torch.testing.assert_close(merged[0],torch.tensor([-2.,-2.]))
        self.assertTrue(stats['conflict'])
        self.assertFalse(stats['projected'])

    def test_projection_geometry_across_tensor_groups(self):
        generator=torch.Generator().manual_seed(407)
        for _ in range(12):
            r=[torch.randn(5,generator=generator,dtype=torch.float64),torch.randn(2,3,generator=generator,dtype=torch.float64)]
            f=[torch.randn(5,generator=generator,dtype=torch.float64),torch.randn(2,3,generator=generator,dtype=torch.float64)]
            merged,stats=self.api.combine_gradients(r,f,enabled=True,min_retain_norm=1e-8)
            dot=sum((rr*(gg-rr)).sum().item() for rr,gg in zip(r,merged))
            self.assertGreaterEqual(dot,-1e-10)
            self.assertTrue(all(torch.isfinite(g).all() for g in merged))

    def test_bad_shapes_and_nonfinite_values_fail(self):
        cases=[([torch.ones(2)],[]),([torch.ones(2)],[torch.ones(3)]),([torch.tensor([float('nan')])],[torch.ones(1)]),([torch.ones(1)],[torch.tensor([float('inf')])])]
        for r,f in cases:
            with self.assertRaises(ValueError): self.api.combine_gradients(r,f,enabled=True,min_retain_norm=1e-8)

    def test_real_sgd_step_preserves_retain_on_conflicting_quadratic(self):
        p=nn.Parameter(torch.tensor([1.,0.]))
        opt=torch.optim.SGD([p],lr=0.1)
        surgery=self.api.SurgeryStep(enabled=True,min_retain_norm=1e-8,diagnostic_interval=1)
        surgery(0.5*p.square().sum(),-2*p[0]+p[1],[p],1)
        opt.step()
        stats=surgery.after_update([p],1)
        torch.testing.assert_close(p,torch.tensor([0.9,-0.1]))
        self.assertAlmostEqual(stats['actual_retain_linear_change'],-0.1,places=6)
        self.assertLess(0.5*p.square().sum().item(),0.5)

    def test_control_two_backward_matches_joint_loss_gradient(self):
        initial=torch.tensor([0.4,-0.2,0.8],dtype=torch.float64)
        p=nn.Parameter(initial.clone())
        (2*p.square().sum()+0.5*(p-1).square().sum()).backward()
        expected=p.grad.clone()
        q=nn.Parameter(initial.clone())
        surgery=self.api.SurgeryStep(enabled=False,min_retain_norm=1e-8,diagnostic_interval=1)
        surgery(2*q.square().sum(),0.5*(q-1).square().sum(),[q],1)
        torch.testing.assert_close(q.grad,expected,rtol=1e-12,atol=1e-12)

    def test_existing_checkpoint_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'existing.pt'; path.write_bytes(b'existing')
            with self.assertRaises(FileExistsError): self.api.save_checkpoint(path,{'weight':torch.ones(1)})
            self.assertEqual(path.read_bytes(),b'existing')

    def test_reference_control_matches_existing_training_and_freezes_prefix(self):
        import S03_reference as original
        reference=importlib.import_module('S04_reference')
        class Backbone(nn.Module):
            def __init__(self):
                super().__init__(); self.blocks=nn.ModuleList([nn.Linear(4,4),nn.Linear(4,4)]); self.norm=nn.LayerNorm(4); self.head=nn.Linear(4,3)
            def forward_features(self,x):
                for b in self.blocks: x=torch.tanh(b(x))
                return self.norm(x)
            def forward_head(self,x,pre_logits=False): return x if pre_logits else self.head(x)
        class Model(nn.Module):
            def __init__(self): super().__init__(); self.backbone=Backbone(); self.head=self.backbone.head
        cfg={'seed':19,'model':{'mo_ckpt':'unused','num_classes':3},'data':{'batch_size':4,'workers':0,'split':'unused','forget':'unused'},'train':{'steps':4,'lr':0.001,'trainable_blocks':1,'lambda_cka_f':2.0},'output':{'save_path':'unused'}}
        batches={'retain':[(torch.arange(16).reshape(4,4).float()/16,torch.tensor([0,1,0,1]))],'forget':[(torch.arange(16).reshape(4,4).float()/17,torch.tensor([2,2,2,2]))]}
        initial={}
        def load(*args):
            model=Model(); initial['state']=copy.deepcopy(model.state_dict()); return model
        with patch.object(original,'load_mo',side_effect=load),patch.object(original,'get_loaders',return_value=batches),patch('torch.cuda.is_available',return_value=False):
            baseline=original.main(config=cfg)
        surgery=self.api.SurgeryStep(enabled=False,min_retain_norm=1e-8,diagnostic_interval=1)
        records=[]
        def after(model,step): records.append(surgery.after_update([p for p in model.parameters() if p.requires_grad],step))
        with patch.object(reference,'load_mo',side_effect=load),patch.object(reference,'get_loaders',return_value=batches),patch('torch.cuda.is_available',return_value=False):
            model=reference.main(config=cfg,gradient_step=surgery,after_step=after)
        self.assertEqual(len(records),4)
        for name,value in baseline.state_dict().items(): torch.testing.assert_close(model.state_dict()[name],value,atol=2e-6,rtol=1e-5)
        for name,value in initial['state'].items():
            if 'blocks.0.' in name: self.assertTrue(torch.equal(value,model.state_dict()[name]))

if __name__=='__main__': unittest.main()
