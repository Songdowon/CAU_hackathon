import unittest

import torch

from unlearn_remap import proximal_l2sp_step


class ProximalL2SpTests(unittest.TestCase):
    def test_applies_reference_centered_pull_after_optimizer_step(self):
        parameter = torch.nn.Parameter(torch.tensor([3.0, -1.0]))
        anchor = torch.tensor([1.0, 1.0])

        proximal_l2sp_step([parameter], [anchor], lr=0.1, strength=2.0)

        torch.testing.assert_close(parameter, torch.tensor([2.6, -0.6]))

    def test_uses_learning_rate_in_pull_scale(self):
        parameter = torch.nn.Parameter(torch.tensor([3.0]))
        anchor = torch.tensor([1.0])

        proximal_l2sp_step([parameter], [anchor], lr=0.01, strength=2.0)

        torch.testing.assert_close(parameter, torch.tensor([2.96]))

    def test_rejects_mismatched_parameter_and_anchor_lists(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        with self.assertRaises(ValueError):
            proximal_l2sp_step([parameter], [], lr=0.1, strength=2.0)

    def test_rejects_pull_that_would_overshoot_anchor(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        with self.assertRaises(ValueError):
            proximal_l2sp_step([parameter], [torch.tensor([0.0])], lr=0.1, strength=11.0)


if __name__ == '__main__':
    unittest.main()
