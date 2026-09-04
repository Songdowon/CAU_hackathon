import copy
import importlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([0.]))
        self.register_buffer('counter', torch.tensor(0, dtype=torch.int64))
        self.drop = nn.Dropout(0.2)

class S03Tests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('S03'), 'S03 averaging implementation is missing')
        self.api = importlib.import_module('S03')
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def tracker(self):
        return self.api.TrajectoryAverages(total_steps=4, ema_start=2, decays={'ema':0.5},
                                          checkpoint_steps=[3,4], snapshot_dir=self.root/'snapshots')

    def paths(self):
        return {k:self.root/(k+'.pt') for k in ['last','avg','ema']}

    def test_tail_average_and_ema_exclude_initial_and_warmup_weights(self):
        tracker, model = self.tracker(), TinyModel()
        for step, value in enumerate([1000., -10., 20., 40.], 1):
            with torch.no_grad(): model.weight.fill_(value); model.counter.fill_(step)
            tracker.observe(model, step)
        report=tracker.export(model, self.paths())
        expected={'last':40.,'avg':30.,'ema':22.5}
        for key, value in expected.items():
            sd=torch.load(self.paths()[key],weights_only=True)['model']
            self.assertAlmostEqual(sd['weight'].item(), value, places=5)
            self.assertEqual(sd['counter'].item(),4)
            TinyModel().load_state_dict(sd,strict=True)
        self.assertEqual(report['ema_updates'],3)
        self.assertEqual(report['checkpoint_steps'],[3,4])
        snap=torch.load(self.root/'snapshots'/'step-000003.pt',weights_only=True)
        self.assertEqual(snap['model']['weight'].item(),20.)

    def test_observation_preserves_source_parameters_gradients_modes_and_rng(self):
        tracker, model = self.tracker(), TinyModel()
        model.weight.grad=torch.tensor([7.])
        before=copy.deepcopy(model.state_dict())
        rng=torch.get_rng_state().clone()
        for step in range(1,5): tracker.observe(model,step)
        for key,value in before.items(): self.assertTrue(torch.equal(value,model.state_dict()[key]))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(model.weight.grad.item(),7.)
        self.assertTrue(model.training and model.drop.training and model.weight.requires_grad)

    def test_missing_or_duplicate_optimizer_steps_are_rejected(self):
        tracker, model = self.tracker(), TinyModel()
        with self.assertRaises(ValueError): tracker.observe(model,2)
        tracker.observe(model,1)
        with self.assertRaises(ValueError): tracker.observe(model,1)
        with self.assertRaises(ValueError): tracker.export(model,self.paths())

    def test_invalid_window_is_rejected_before_creating_artifacts(self):
        with self.assertRaises(ValueError):
            self.api.TrajectoryAverages(total_steps=4,ema_start=0,decays={'ema':1.0},
                                       checkpoint_steps=[3,3],snapshot_dir=self.root/'snapshots')
        self.assertFalse((self.root/'snapshots').exists())

    def test_existing_checkpoint_is_preserved(self):
        target=self.root/'existing.pt'
        target.write_bytes(b'existing experiment')
        with self.assertRaises(FileExistsError): self.api.save_checkpoint(target,TinyModel().state_dict())
        self.assertEqual(target.read_bytes(),b'existing experiment')

    def test_batchnorm_is_rejected_instead_of_silently_using_wrong_statistics(self):
        tracker=self.tracker()
        with self.assertRaises(ValueError): tracker.observe(nn.BatchNorm1d(4),1)

    def test_callback_observes_post_update_weights_without_changing_training(self):
        import S02_reference as original
        reference=importlib.import_module('S03_reference')
        class Backbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks=nn.ModuleList([nn.Linear(4,4),nn.Linear(4,4)])
                self.norm=nn.LayerNorm(4)
                self.head=nn.Linear(4,3)
            def forward_features(self,x):
                for b in self.blocks: x=torch.tanh(b(x))
                return self.norm(x)
            def forward_head(self,x,pre_logits=False): return x if pre_logits else self.head(x)
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone=Backbone()
                self.head=self.backbone.head
        cfg={'seed':19,'model':{'mo_ckpt':'unused','num_classes':3},
             'data':{'batch_size':4,'workers':0,'split':'unused','forget':'unused'},
             'train':{'steps':4,'lr':0.001,'trainable_blocks':1,'lambda_cka_f':2.0},
             'output':{'save_path':str(self.root/'baseline.pt')}}
        batches={'retain':[(torch.arange(16).reshape(4,4).float()/16,torch.tensor([0,1,0,1]))],
                 'forget':[(torch.arange(16).reshape(4,4).float()/17,torch.tensor([2,2,2,2]))]}
        import yaml
        config_path=self.root/'config.yaml'
        config_path.write_text(yaml.safe_dump(cfg))
        def load(*args): return Model()
        with patch.object(original,'load_mo',side_effect=load), patch.object(original,'get_loaders',return_value=batches), patch('torch.cuda.is_available',return_value=False), patch('sys.argv',['test','--config',str(config_path)]):
            original.main()
        tracker=self.tracker()
        with patch.object(reference,'load_mo',side_effect=load), patch.object(reference,'get_loaders',return_value=batches), patch('torch.cuda.is_available',return_value=False):
            model=reference.main(config=cfg,after_step=tracker.observe)
        original_sd=torch.load(self.root/'baseline.pt',weights_only=True)['model']
        for name,value in original_sd.items(): self.assertTrue(torch.equal(value,model.state_dict()[name]),name)
        checkpoint=torch.load(self.root/'snapshots'/'step-000004.pt',weights_only=True)['model']
        for name,value in original_sd.items(): self.assertTrue(torch.equal(value,checkpoint[name]),name)

if __name__=='__main__': unittest.main()
