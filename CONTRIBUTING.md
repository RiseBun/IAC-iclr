# Contributing

This repository is the public IAC evaluation package. Please keep changes
aligned with the frozen protocol:

1. Do not add raw NAVSIM/Waymo frames, private GT, generated videos or WAM
   checkpoints.
2. Primary motion fields stay lateral speed, yaw rate and curvature unless a
   new protocol revision re-admits longitudinal metric scale with a frozen
   error budget.
3. Unsupported capabilities must remain `unavailable`, never zero-filled.
4. Prefer extending the public entrypoints in `scripts/README.md` over adding
   model-specific runners to the default path.

```bash
python -m pip install -e .
PYTHONPATH=src:. python -m pytest -q
sha256sum -c weights/SHA256SUMS.txt
```
