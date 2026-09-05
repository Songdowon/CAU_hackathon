import json
from pathlib import Path
import tempfile
import unittest

import torch
import yaml

from S13 import merge_state_dict, run_from_config


class S13ConsensusMergeTests(unittest.TestCase):
    def test_opposing_helpers_cannot_cancel_anchor_delta(self):
        base = {"w": torch.tensor([0.0, 0.0, 0.0])}
        anchor = {"w": torch.tensor([2.0, -2.0, 0.0])}
        helpers = [
            {"w": torch.tensor([1.0, 5.0, 9.0])},
            {"w": torch.tensor([3.0, -3.0, -9.0])},
        ]

        merged, audit = merge_state_dict(base, anchor, helpers)
        delta = merged["w"] - base["w"]
        anchor_delta = anchor["w"] - base["w"]

        self.assertTrue(torch.all(torch.sign(delta[:2]) == torch.sign(anchor_delta[:2])))
        self.assertEqual(delta[2].item(), 0.0)
        self.assertAlmostEqual(delta.norm().item(), anchor_delta.norm().item(), places=6)
        self.assertEqual(audit["summary"]["changed_float_tensors"], 1)

    def test_head_and_frozen_tensors_are_copied_from_anchor(self):
        base = {
            "backbone.head.weight": torch.tensor([[0.0, 0.0]]),
            "backbone.blocks.0.weight": torch.tensor([4.0]),
        }
        anchor = {
            "backbone.head.weight": torch.tensor([[9.0, 8.0]]),
            "backbone.blocks.0.weight": torch.tensor([4.0]),
        }
        helpers = [{
            "backbone.head.weight": torch.tensor([[1.0, 2.0]]),
            "backbone.blocks.0.weight": torch.tensor([7.0]),
        }]

        merged, audit = merge_state_dict(base, anchor, helpers)

        self.assertTrue(torch.equal(merged["backbone.head.weight"], anchor["backbone.head.weight"]))
        self.assertTrue(torch.equal(merged["backbone.blocks.0.weight"], anchor["backbone.blocks.0.weight"]))
        self.assertEqual(audit["summary"]["head_tensors_copied"], 1)
        self.assertEqual(audit["summary"]["frozen_tensors_copied"], 1)

    def test_tiny_numeric_drift_is_restored_to_base(self):
        base = {"w": torch.tensor([1.0, -1.0])}
        anchor = {"w": torch.tensor([1.0 + 5e-6, -1.0 - 5e-6])}
        helpers = [{"w": torch.tensor([1.0 - 4e-6, -1.0 + 4e-6])}]

        merged, audit = merge_state_dict(base, anchor, helpers, numerical_delta_max=1e-5)

        self.assertTrue(torch.equal(merged["w"], base["w"]))
        self.assertEqual(audit["summary"]["numerical_frozen_tensors_copied"], 1)

    def test_incompatible_state_dict_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "keys"):
            merge_state_dict(
                {"w": torch.tensor([0.0])},
                {"w": torch.tensor([1.0])},
                [{"other": torch.tensor([1.0])}],
            )

    def test_config_run_writes_cpu_checkpoint_and_audit(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            base_path = root / "base.pt"
            anchor_path = root / "anchor.pt"
            helper_path = root / "helper.pt"
            out_path = root / "S13.pt"
            audit_path = root / "S13.audit.json"
            torch.save({"model": {"w": torch.tensor([0.0, 0.0])}}, base_path)
            torch.save({"model": {"w": torch.tensor([2.0, -2.0])}}, anchor_path)
            torch.save({"model": {"w": torch.tensor([1.0, -3.0])}}, helper_path)
            config_path = root / "S13.yaml"
            config_path.write_text(yaml.safe_dump({
                "base": str(base_path),
                "anchor": str(anchor_path),
                "helpers": [str(helper_path)],
                "output": str(out_path),
                "audit": str(audit_path),
            }))

            run_from_config(config_path)

            payload = torch.load(out_path, map_location="cpu", weights_only=True)
            self.assertEqual(set(payload), {"model"})
            self.assertTrue(all(t.device.type == "cpu" for t in payload["model"].values()))
            saved_audit = json.loads(audit_path.read_text())
            self.assertEqual(saved_audit["method"], "anchor_preserving_consensus")
            self.assertEqual(saved_audit["sources"]["anchor"], str(anchor_path))


if __name__ == "__main__":
    unittest.main()
