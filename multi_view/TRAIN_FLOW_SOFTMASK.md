# Training Flow (Soft Mask Ref)

Detailed ASCII flow for per-frame soft mask on ref latents.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                               DATASET LAYER                                  │
└──────────────────────────────────────────────────────────────────────────────┘
   Video (T frames)                           Ref images (R)
   ┌──────────────────────┐                   ┌──────────────────────────┐
   │ raw video frames     │                   │ precomputed face crops   │
   │ (PIL list)           │                   │ (PIL list)                │
   └──────────┬───────────┘                   └──────────┬───────────────┘
              │                                          │
              │ sample_ids (absolute frame indices)      │ parse ref filename
              │ e.g., [min...max]                        │ e.g., 199_crop_face.png
              v                                          v
   ┌──────────────────────┐                   ┌──────────────────────────┐
   │ video_frame_indices  │                   │ ref_frame_indices        │
   │(absolute, with offset)│                  │ (absolute, with offset)  │
   └──────────┬───────────┘                   └──────────┬───────────────┘
              │                                          │
              │                                          │
              v                                          v
   ┌──────────────────────┐                   ┌──────────────────────────┐
   │ video PIL list       │                   │ ref PIL list             │
   └──────────┬───────────┘                   └──────────┬───────────────┘
              │                                          │
              └──────────────────┬───────────────────────┘
                                 v
                    (batch dict passed into pipeline)

┌──────────────────────────────────────────────────────────────────────────────┐
│                             PIPELINE PREPROCESS                              │
└──────────────────────────────────────────────────────────────────────────────┘
   video PIL list                               ref PIL list
   ┌──────────────────────┐                     ┌────────────────────────┐
   │ preprocess_video     │                     │ extrac_ref_latents     │
   │ (B,C,T,H,W in [-1,1])│                     │ (B,C,R,H,W)            │
   └──────────┬───────────┘                     └──────────┬─────────────┘
              │                                          │
              v                                          v
   ┌──────────────────────┐                     ┌────────────────────────┐
   │ input_latents         │                     │ ref_images_latents     │
   │ B,C,F,H',W'           │                     │ B,C,R,H',W'            │
   │ F=(T-1)//4 + 1        │                     │ (R=ref_num)            │
   └──────────┬───────────┘                     └──────────┬─────────────┘
              │                                          │
              └──────────────────┬───────────────────────┘
                                 v

┌──────────────────────────────────────────────────────────────────────────────┐
│                       PER-FRAME SOFT MASK (NEW)                               │
└──────────────────────────────────────────────────────────────────────────────┘
Inputs:
  - input_latents (B,C,F,H',W')
  - ref_images_latents (B,C,R,H',W')
  - video_frame_indices (absolute)
  - ref_frame_indices (absolute)

Step 1: map video frames to latent frames
  latent_vid_idx = video_frame_indices[::stride]
  stride = time_division_factor (default 4)

Step 2: for each latent frame t
  nearest_ref = argmin_r |latent_vid_idx[t] - ref_frame_indices[r]|
  weights[t, r] = 0.3 for nearest_ref, else 1.0
  weights[t, :] = weights[t, :] / sum(weights[t, :])

Step 3: weighted sum to build per-frame ref condition
  ref_cond[t] = sum_r weights[t, r] * ref_latent[r]

Step 4: inject into video latents
  input_latents = input_latents + ref_cond
  ref_images_latents = None
  num_ref_images = 0

┌──────────────────────────────────────────────────────────────────────────────┐
│                               MAIN MODEL                                     │
└──────────────────────────────────────────────────────────────────────────────┘
   input_latents + noise  ──► scheduler ──► DiT/WanModel ──► loss ──► backprop

Notes:
- ref tokens are removed from self-attention (ref_images_latents=None)
  to avoid a global copy-paste shortcut.
- offset is preserved because indices are absolute frame IDs.
```
