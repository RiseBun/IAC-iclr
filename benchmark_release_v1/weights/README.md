# Included optical-flow weights

The release includes the two image-side checkpoints used by the Level-1
baseline and its challenger:

- `raft_large_C_T_SKHT_V2-ff5fadd5.pth`: torchvision RAFT-Large
  `C_T_SKHT_V2` weights. Upstream:
  https://download.pytorch.org/models/raft_large_C_T_SKHT_V2-ff5fadd5.pth
- `sea_raft_model.safetensors`: the SEA-RAFT checkpoint used in the IAC
  experiments. The implementation and upstream license are maintained at:
  https://github.com/princeton-vl/SEA-RAFT

Verify both files with `SHA256SUMS.txt` before running an experiment. These are
third-party weights and remain subject to their upstream licenses.
