import copy
import importlib
import math
import random
import unittest
import numpy as np
import torch
from torch import nn

try:
    s02 = importlib.import_module("S02")
except ModuleNotFoundError as error:
    if error.name != "S02":
        raise
    s02 = None

class TinyViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.patch_embed = nn.Linear(4, 4)
        self.backbone.blocks = nn.ModuleList([nn.Sequential(nn.Linear(4, 4), nn.Tanh(), nn.Dropout(.2)) for _ in range(4)])
        self.backbone.norm = nn.LayerNorm(4)
        self.backbone.head = nn.Linear(4, 3)
    def forward(self, x):
        x = self.backbone.patch_embed(x)
        for block in self.backbone.blocks:
            x = x + block(x)
        return self.backbone.head(self.backbone.norm(x))

class LayerSelectionTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(s02, "S02 implementation is missing")
        torch.manual_seed(19)
        self.model = TinyViT()

    def test_noncontiguous_selection_updates_early_block_and_freezes_others(self):
        params = s02.apply_selection(self.model, [0, 3], train_norm=True, train_head=True)
        before = copy.deepcopy(self.model.state_dict())
        loss = nn.functional.cross_entropy(self.model(torch.randn(8, 4)), torch.arange(8) % 3)
        loss.backward()
        torch.optim.SGD(params, lr=.1).step()
        after = self.model.state_dict()
        self.assertFalse(torch.equal(before['backbone.blocks.0.0.weight'], after['backbone.blocks.0.0.weight']))
        for key in before:
            if key.startswith(('backbone.patch_embed.', 'backbone.blocks.1.', 'backbone.blocks.2.')):
                self.assertTrue(torch.equal(before[key], after[key]), key)

    def test_head_only_freezes_norm_and_all_blocks(self):
        params = s02.apply_selection(self.model, [], train_norm=False, train_head=True)
        actual = [name for name, p in self.model.named_parameters() if p.requires_grad]
        self.assertEqual(actual, ['backbone.head.weight', 'backbone.head.bias'])
        self.assertEqual(len(params), 2)

    def test_invalid_masks_are_rejected(self):
        for blocks in ([4], [-1], [1, 1]):
            with self.assertRaises(ValueError):
                s02.apply_selection(self.model, blocks, train_norm=True, train_head=True)
        with self.assertRaises(ValueError):
            s02.apply_selection(self.model, [], train_norm=False, train_head=False)

    def test_ratio_of_mean_norms_and_low_signal_guard(self):
        row = s02.summarize_norms([2., 8.], [1., 3.], epsilon=.5, min_norm=1e-8)
        self.assertAlmostEqual(row['ratio'], 2.0)
        self.assertTrue(row['eligible'])
        low = s02.summarize_norms([2., 3.], [0., 0.], epsilon=1e-8, min_norm=1e-8)
        self.assertFalse(low['eligible'])
        self.assertTrue(math.isfinite(low['ratio']))
        with self.assertRaises(ValueError):
            s02.summarize_norms([float('nan')], [1.], epsilon=1e-8, min_norm=1e-8)

    def test_ranking_uses_only_eligible_blocks_and_keeps_budget(self):
        rows = [dict(group='block.0', ratio=2., eligible=True), dict(group='block.1', ratio=9., eligible=False), dict(group='block.2', ratio=5., eligible=True), dict(group='block.3', ratio=3., eligible=True), dict(group='head', ratio=1000., eligible=True)]
        self.assertEqual(s02.choose_blocks(rows, 2), [2, 3])
        with self.assertRaises(ValueError):
            s02.choose_blocks(rows, 4)

    def test_probe_is_repeatable_and_does_not_change_model_modes_flags_or_rng(self):
        x = torch.randn(8, 4)
        batches = {'forget': [(x, torch.arange(8) % 3)] * 2, 'retain': [(x * .7, (torch.arange(8) + 1) % 3)] * 2}
        self.model.backbone.blocks[1].eval()
        self.model.backbone.blocks[0][0].weight.requires_grad_(False)
        state = copy.deepcopy(self.model.state_dict())
        modes = [m.training for m in self.model.modules()]
        flags = [p.requires_grad for p in self.model.parameters()]
        random.seed(72)
        np.random.seed(72)
        cpu_rng = torch.get_rng_state().clone()
        py_rng = random.getstate()
        np_rng = np.random.get_state()
        rows1 = s02.probe_gradient_ratios(self.model, lambda: batches, batches=2, seed=42)
        rows2 = s02.probe_gradient_ratios(self.model, lambda: batches, batches=2, seed=42)
        self.assertEqual(rows1, rows2)
        self.assertEqual(len(rows1), 6)
        self.assertTrue(all(math.isfinite(r['ratio']) for r in rows1))
        self.assertTrue(torch.equal(cpu_rng, torch.get_rng_state()))
        self.assertEqual(py_rng, random.getstate())
        self.assertTrue(np.array_equal(np_rng[1], np.random.get_state()[1]))
        self.assertEqual(modes, [m.training for m in self.model.modules()])
        self.assertEqual(flags, [p.requires_grad for p in self.model.parameters()])
        self.assertTrue(all(torch.equal(v, self.model.state_dict()[k]) for k, v in state.items()))
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_last_blocks_mask_matches_teammate_baseline(self):
        import unlearn_remap
        other = copy.deepcopy(self.model)
        unlearn_remap.set_trainable(other, 2)
        s02.apply_selection(self.model, [2, 3], train_norm=True, train_head=True)
        self.assertEqual([n for n,p in other.named_parameters() if p.requires_grad], [n for n,p in self.model.named_parameters() if p.requires_grad])

    def test_metadata_reservation_preserves_an_existing_run(self):
        import json
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / 'selection.json'
            s02.write_metadata(report, {'status': 'preparing'}, exclusive=True)
            with self.assertRaises(FileExistsError):
                s02.write_metadata(report, {'status': 'replacement'}, exclusive=True)
            self.assertEqual(json.loads(report.read_text()), {'status': 'preparing'})

if __name__ == '__main__':
    unittest.main()
