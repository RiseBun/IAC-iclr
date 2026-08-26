# DriveWAM Action-Image Diagnostic

## Scope

This is a NAVSIM native-data intervention test with 20 scenes and three
branches per scene: `logged`, `left`, and `right`. Each branch uses the same
four native history frames and an independently generated future video. The
realized future ego state and NAVSIM PDM task score are kept separate from the
generated images.

## Results

| Protocol | IAC coverage | Mean realized-state compatibility | Mean response alignment | FCS @ 0.70 |
|---|---:|---:|---:|---:|
| Native real future-image control | 1.00 | 0.5454 | 0.0000 | 1.00 (21/60 compatible) |
| DriveWAM generated, 4/4 steps, chunk 0 | 1.00 | 0.0876 | 0.0003 | null (0/60 compatible) |
| DriveWAM generated, 1/1 steps, chunk 0 | 1.00 | 0.0891 | 0.0212 | null (0/60 compatible) |
| DriveWAM generated, 4/4 steps, chunk 1 (3-scene probe) | 1.00 | 0.0131 | 0.0024 | null (0/9 compatible) |
| DriveWAM generated, 4/4 steps, chunk 1 | 1.00 | 0.0876 | 0.0220 | null (0/60 compatible) |

All generated branches report `action_injection_verified=true`. The 4/4 run
does not improve over the 1/1 run, and moving the injected pose to the future
action slot does not restore action-conditioned image response.

## Direct image-sensitivity audit

For the 20-scene 4/4 run, all 60 branches were paired at each of four future
times. The mean pairwise image MAE between different action branches is
`0.000831` after normalization to `[0, 1]`; only `0.00115` of pixels change by
more than 2/255 on average. This is effectively a branch-invariant video
under the tested intervention, despite verified action injection.

## Interpretation

The current evidence does not support “IAC alone is the primary failure.” The
native-image control establishes that the same IAC can extract a plausible
trajectory support from real NAVSIM future images. The much lower compatibility
and zero FCS-compatible branches for DriveWAM outputs are dominated by weak
action-to-image response in this DriveWAM checkpoint/configuration. IAC still
has a non-zero error floor, so it must be calibrated and reported separately,
but improving IAC cannot recover information absent from the generated frames.

## Reproduction artifacts on server

- `work_dirs/drivewam_generated_closed_loop_20_v4_final.json`
- `work_dirs/drivewam_generated_closed_loop_20_v4_sensitivity.json`
- `work_dirs/drivewam_generated_closed_loop_20_final.json`
- `work_dirs/navsim_native_image_control_20_final.json`
- `work_dirs/drivewam_generated_closed_loop_3_v4_c1_final.json`

The next model-side test should use a WAM/checkpoint whose public inference
path explicitly conditions future video on the action, or expose the model's
native action-conditioning API. The IAC side should then be evaluated with the
same native-image control and a LiDAR/geometry oracle upper-bound split.
