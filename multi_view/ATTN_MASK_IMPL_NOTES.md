# Attention Mask Implementation Notes (Per‑Frame Soft Mask)

This document summarizes the **current implementation** of the per‑frame soft attention mask:
- where indices are introduced
- how the nearest ref is computed
- how the full `S×S` mask is built
- and how it is injected into Self‑Attention

---

## 1) Data indices: where they come from

### A) `videodataset.py`
- **Video indices**: `sample_ids` (absolute frame indices) are produced by `np.linspace(...)` when sampling the 5‑second segment.
- **Ref indices**: parsed from filename like `199_crop_face.png` using regex.

Returned fields:
- `video_frame_indices = sample_ids`
- `ref_frame_indices = [frame_id_from_filename, ...]`

### B) `videodataset_movie.py`
- **Video indices**: `sample_ids` are filled either by `np.linspace(...)` (normal case) or by sequential frames + padding when video is short.
- **Ref indices**: parsed from ref filename using regex `r"(\d+)"`.

Returned fields:
- `video_frame_indices = sample_ids`
- `ref_frame_indices = [...]`

---

## 2) Training input plumbing

In `train.py`, the indices are passed into the pipeline:

```python
"ref_frame_indices": [d.get("ref_frame_indices") for d in data],
"video_frame_indices": [d.get("video_frame_indices") for d in data],
"ref_soft_mask_strength": args.ref_soft_mask_strength,
```

Also added config:

```yaml
train_args:
  ref_soft_mask_strength: 0.3
```

---

## 3) How nearest ref is computed (per latent frame)

Location: `WanVideoUnit_RefFrameSoftMask` in `diffsynth/pipelines/wan_video_new.py`

Key steps:
1) **Map video indices to latent indices** (time stride)
   ```python
   stride = pipe.time_division_factor or 4
   latent_vid_idx = video_frame_indices[::stride]
   latent_vid_idx = pad_or_trim(latent_vid_idx, f_len)
   ```
2) **Nearest ref per latent frame**
   ```python
   dist = |latent_vid_idx[t] - ref_frame_indices[r]|
   nearest = argmin_r dist
   ```

---

## 4) How the full `S×S` mask is built

Let:
- `F = latent video frames`
- `R = ref images`
- `tokens_per_frame = (H' / patch_h) * (W' / patch_w)`
- `S = (F + R) * tokens_per_frame`

We create:
```python
attn_mask = zeros((B, S, S), dtype=latents.dtype)
```

For each latent frame `t`, let `r = nearest[t]`:
- rows corresponding to **video tokens of frame t**
- columns corresponding to **ref tokens of frame r**

We apply:
```python
mask_value = log(strength)   # strength = 0.3 by default
attn_mask[b, row_start:row_end, col_start:col_end] = mask_value
```

This is a **soft mask** (log-scale add to attention logits).

---

## 5) Inject mask into Self‑Attention

### A) `model_fn_wan_video`
- accepts `self_attn_mask`
- passes it into each DiT block

### B) `DiTBlock -> SelfAttention`
- mask is expanded to `(B, 1, S, S)`
- passed into `F.scaled_dot_product_attention(q,k,v, attn_mask=...)`

---

## 6) ASCII visualization

```
Data (absolute indices)
┌───────────────────────────────────────────────┐
│ video_frame_indices = [ .. absolute ids .. ] │
│ ref_frame_indices   = [ .. absolute ids .. ] │
└───────────────────────────────────────────────┘
                 │
                 ▼
Latent time alignment
┌───────────────────────────────────────────────┐
│ latent_vid_idx = video_frame_indices[::4]    │
│ (pad/trim to length F)                       │
└───────────────────────────────────────────────┘
                 │
                 ▼
Nearest ref per latent frame
┌───────────────────────────────────────────────┐
│ for each t: nearest = argmin |vid[t]-ref[r]| │
└───────────────────────────────────────────────┘
                 │
                 ▼
Build mask blocks (soft)
┌───────────────────────────────────────────────┐
│ S = (F+R) * tokens_per_frame                 │
│ attn_mask[B, S, S] initialized to 0          │
│ for each t:                                   │
│   mask block (video frame t -> ref r)        │
│   add log(0.3)                               │
└───────────────────────────────────────────────┘
                 │
                 ▼
Self-Attention
┌───────────────────────────────────────────────┐
│ scaled_dot_product_attention(..., attn_mask) │
└───────────────────────────────────────────────┘
```

---

## 7) Runtime mask stats print

On first batch, the pipeline prints:
```
[RefMask] attn_mask shape=(B,S,S) dtype=... size~X.XXGB tokens_per_frame=... frames(F)=... ref=... strength=...
```

---

## Files touched

- `datasets/videodataset.py`
- `datasets/videodataset_movie.py`
- `train.py`
- `conf/multi-view.yaml`
- `diffsynth/pipelines/wan_video_new.py`
- `diffsynth/models/wan_video_dit.py`

