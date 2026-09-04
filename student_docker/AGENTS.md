# Experiment organization

- The experiment line beginning with S01.py uses uppercase IDs S01, S02, S03, and so on. S02 is the current layer-selection study; S02-* are its ablations. The next independent experiment is S03.
- Preserve the teammate's existing lowercase r001, r002, ... experiments and their files.
- Read S_EXPERIMENTS.md for the S-line naming convention, existing results, and experiment record format.
- Keep each S ID aligned across its script (S02.py), config (configs/S02.yaml), checkpoint (models/S02.pt), logs, and result tags.
- Put the script name in each S config (script: S02.py) so tools/run_exp.py uses the correct implementation.
- Preserve the restored unlearn.py competition template. Put experimental implementations in their S files.
- Preserve previous checkpoints/results; distinguish reruns with a timestamp or run identifier.
- When training is requested, use the existing GPU-locking runner tools/run_exp.py because the GPU is shared.
- Keep pending/running experiments separate from verified completed results. Public validation scores are not private leaderboard scores.
