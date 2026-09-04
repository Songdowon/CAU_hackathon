import importlib
from pathlib import Path
import unittest

import torch
import yaml


class S08Tests(unittest.TestCase):
    def module(self):
        try:
            return importlib.import_module("S08")
        except ModuleNotFoundError:
            self.fail("S08 Fisher-weighted anchoring is not implemented")

    def test_last_blocks_scope_excludes_head(self):
        s = self.module()

        class Backbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(2)])
                self.norm = torch.nn.LayerNorm(2)
                self.head = torch.nn.Linear(2, 2)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = Backbone()

        selected = dict(s.select_anchor_parameters(Model(), "last_6_blocks"))
        self.assertTrue(selected)
        self.assertTrue(all(name.startswith("backbone.blocks.") for name in selected))
        self.assertFalse(any("head" in name or "norm" in name for name in selected))

    def test_fisher_normalization_is_per_layer(self):
        s = self.module()
        fisher = {
            "backbone.blocks.6.attn.weight": torch.tensor([1.0, 3.0]),
            "backbone.blocks.6.mlp.weight": torch.tensor([2.0]),
            "backbone.blocks.7.attn.weight": torch.tensor([4.0, 4.0]),
            "backbone.blocks.8.attn.weight": torch.zeros(2),
        }
        normalized = s.normalize_fisher_per_layer(fisher)
        self.assertTrue(torch.allclose(normalized["backbone.blocks.6.attn.weight"], torch.tensor([0.5, 1.5])))
        self.assertTrue(torch.allclose(normalized["backbone.blocks.6.mlp.weight"], torch.tensor([1.0])))
        self.assertTrue(torch.allclose(normalized["backbone.blocks.7.attn.weight"], torch.ones(2)))
        self.assertEqual(normalized["backbone.blocks.8.attn.weight"].sum().item(), 0.0)

    def test_anchor_loss_is_weighted_sum(self):
        s = self.module()
        p = torch.nn.Parameter(torch.tensor([2.0, 4.0]))
        named = [("backbone.blocks.6.weight", p)]
        anchor = {"backbone.blocks.6.weight": torch.tensor([1.0, 2.0])}
        fisher = {"backbone.blocks.6.weight": torch.tensor([1.0, 3.0])}
        loss = s.fisher_anchor_loss(named, anchor, fisher)
        self.assertAlmostEqual(loss.item(), 13.0)
        loss.backward()
        self.assertTrue(torch.allclose(p.grad, torch.tensor([2.0, 12.0])))

    def test_fisher_estimation_is_nonnegative_and_preserves_weights(self):
        s = self.module()
        model = torch.nn.Linear(2, 2)
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        batches = [(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), torch.tensor([0, 1]))]
        named = list(model.named_parameters())
        fisher = s.estimate_diagonal_fisher(
            model, batches, named, max_samples=2, device=torch.device("cpu"),
            normalize_per_layer=False,
        )
        self.assertEqual(set(fisher), set(before))
        self.assertTrue(all(torch.all(value >= 0) for value in fisher.values()))
        self.assertTrue(any(torch.any(value > 0) for value in fisher.values()))
        for name, p in model.named_parameters():
            self.assertTrue(torch.equal(p.detach(), before[name]))
            self.assertIsNone(p.grad)

    def test_config_is_wired_to_s08(self):
        cfg = yaml.safe_load(Path("configs/S08.yaml").read_text())
        self.assertEqual(cfg["script"], "S08.py")
        self.assertEqual(cfg["output"]["save_path"], "models/S08.pt")
        self.assertEqual(cfg["train"]["fisher_scope"], "last_6_blocks")
        self.assertEqual(cfg["train"]["fisher_samples"], 4096)
        self.assertTrue(cfg["train"]["normalize_fisher_per_layer"])
        self.assertTrue(cfg["train"]["freeze_norm"])
        self.assertGreater(cfg["train"]["lambda_anchor"], 0)

    def test_config_preserves_matched_s06_seed0_objective(self):
        cfg = yaml.safe_load(Path("configs/S08.yaml").read_text())
        base = yaml.safe_load(Path("configs/s06_seed0.yaml").read_text())
        for key, value in base["train"].items():
            self.assertEqual(cfg["train"].get(key), value, key)
        self.assertEqual(cfg["seed"], base["seed"])



if __name__ == "__main__":
    unittest.main()
