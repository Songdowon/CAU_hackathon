# baselines/

This directory is shipped in the Student image as a deliberately weak, runnable
example for exercising the train -> validate -> submit pipeline end to end.

`ga_example.py`: a minimal, deliberately-weak GA (gradient ascent) example
extracted from `unlearn_baselines.py`, ported to this competition's
`utils.data.get_loaders` and `imagenet_vit.ViTWrapper` interfaces.

It is intentionally weak, so it demonstrates the required shape (load M_o,
touch the forget set, save a checkpoint matching `validate_submission.py`)
without handing participants a competitive submission. It updates all model
parameters, not only the classifier head. Competitive organizer baselines are
not included in the student environment.
