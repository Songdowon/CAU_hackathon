import unittest

from S11 import StateTriggeredCosineTail


class StateTriggeredCosineTailTests(unittest.TestCase):
    def make_controller(self, **overrides):
        kwargs = dict(
            total_steps=10,
            base_lr=3e-5,
            final_lr=3e-6,
            floor=0.05,
            min_step=4,
            window=3,
            off_ratio_threshold=2 / 3,
            ema_beta=0.98,
        )
        kwargs.update(overrides)
        return StateTriggeredCosineTail(**kwargs)

    def test_does_not_trigger_before_minimum_step(self):
        controller = self.make_controller()
        for step in range(4):
            lr, triggered = controller.update(step, raw_cka_f=0.01)
            self.assertFalse(triggered)
            self.assertEqual(lr, 3e-5)
        self.assertIsNone(controller.trigger_step)

    def test_triggers_when_recent_off_ratio_reaches_threshold(self):
        controller = self.make_controller()
        values = [0.20, 0.20, 0.01, 0.01, 0.20]
        for step, value in enumerate(values):
            lr, triggered = controller.update(step, raw_cka_f=value)
        self.assertTrue(triggered)
        self.assertEqual(controller.trigger_step, 4)
        self.assertAlmostEqual(controller.off_ratio, 2 / 3)
        self.assertAlmostEqual(lr, 3e-5)

    def test_reaches_final_lr_on_last_update_after_trigger(self):
        controller = self.make_controller()
        for step in range(5):
            controller.update(step, raw_cka_f=0.01)
        lr, triggered = controller.update(9, raw_cka_f=0.20)
        self.assertFalse(triggered)
        self.assertAlmostEqual(lr, 3e-6)

    def test_never_decays_when_state_does_not_trigger(self):
        controller = self.make_controller()
        for step in range(10):
            lr, triggered = controller.update(step, raw_cka_f=0.20)
            self.assertFalse(triggered)
            self.assertEqual(lr, 3e-5)
        self.assertIsNone(controller.trigger_step)

    def test_does_not_trigger_on_final_update_when_no_tail_remains(self):
        controller = self.make_controller(
            total_steps=5, min_step=0, window=2, off_ratio_threshold=0.5)
        for step in range(4):
            controller.update(step, raw_cka_f=0.20)
        lr, triggered = controller.update(4, raw_cka_f=0.01)
        self.assertFalse(triggered)
        self.assertEqual(lr, 3e-5)
        self.assertIsNone(controller.trigger_step)


if __name__ == '__main__':
    unittest.main()
