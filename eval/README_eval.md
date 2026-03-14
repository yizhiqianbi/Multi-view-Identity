# Evaluation Metrics (IC2 / MD)

This folder provides a simple CLI that implements the metrics defined in `eval.md`:

- **IC2 (Identity Consistency)**: For each frame in the generated video, compute the maximum cosine similarity between its face embedding and the reference image embeddings, then average over frames.
- **MD (Motion Dynamic)**: Estimate head pose angles for each frame, compute per-frame L2 distance between generated and condition angles, then average (lower is better).

## Install

```bash
pip install -r requirements-eval.txt
```

> The first run of InsightFace may download model weights into the model root.
> If you plan to use CurricularFace, install PyTorch for your CUDA/CPU environment.

## Usage

```bash
python eval_metrics.py \
  --gen /path/to/generated.mp4 \
  --cond /path/to/condition.mp4 \
  --refs /path/to/ref_dir \
  --out /tmp/metrics.json
```

### CurricularFace backend (optional)

`--face-backend curricularface` uses the IR backbone from the local `CurricularFace/` repo and applies the same
`l2_norm` normalization as the original implementation.

```bash
python eval_metrics.py \
  --face-backend curricularface \
  --curricularface-repo ./CurricularFace \
  --curricularface-backbone IR_101 \
  --curricularface-ckpt /path/to/ir101.pth \
  --gen /path/to/generated.mp4 \
  --refs /path/to/ref_dir
```

### Notes

- `--gen` and `--cond` can be a video file or a directory of frames.
- `--refs` can be multiple files or directories.
- If you want to use a specific face model, set `--face-model-name` and `--face-model-root`.
- Missing faces are skipped by default (`--missing-policy skip`).

## Example Output

```json
{
  "identity_consistency": {
    "ic2": 0.78,
    "backend": "insightface",
    "total_frames": 120,
    "used_frames": 110,
    "missing_frames": 10
  },
  "motion_dynamic": {
    "md": 3.25,
    "total_pairs": 120,
    "used_pairs": 108,
    "missing_pairs": 12
  }
}
```
