import unittest

from S09 import cosine_tail_lr


class CosineTailLrTests(unittest.TestCase):
    def test_keeps_base_lr_through_tail_start(self):
        for step in (0, 4799, 4800):
            self.assertAlmostEqual(
                cosine_tail_lr(
                    step, total_steps=7200, tail_start=4800,
                    base_lr=3e-5, final_lr=3e-6,
                ),
                3e-5,
            )

    def test_reaches_requested_final_lr_on_last_update(self):
        self.assertAlmostEqual(
            cosine_tail_lr(
                7199, total_steps=7200, tail_start=4800,
                base_lr=3e-5, final_lr=3e-6,
            ),
            3e-6,
        )

    def test_midpoint_is_halfway_between_endpoints(self):
        self.assertAlmostEqual(
            cosine_tail_lr(
                6, total_steps=9, tail_start=4,
                base_lr=3e-5, final_lr=0.0,
            ),
            1.5e-5,
        )

    def test_rejects_invalid_boundaries(self):
        invalid = [
            dict(step=-1, total_steps=7200, tail_start=4800, base_lr=3e-5, final_lr=0.0),
            dict(step=7200, total_steps=7200, tail_start=4800, base_lr=3e-5, final_lr=0.0),
            dict(step=0, total_steps=7200, tail_start=7199, base_lr=3e-5, final_lr=0.0),
            dict(step=0, total_steps=7200, tail_start=4800, base_lr=3e-5, final_lr=4e-5),
        ]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    cosine_tail_lr(**kwargs)


if __name__ == '__main__':
    unittest.main()
