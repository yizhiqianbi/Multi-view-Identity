#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_frames(path, stride=1, max_frames=None):
    if os.path.isdir(path):
        paths = []
        for ext in IMAGE_EXTS:
            paths.extend(Path(path).glob(f"*{ext}"))
            paths.extend(Path(path).glob(f"*{ext.upper()}"))
        for p in sorted(set(paths)):
            img = cv2.imread(str(p))
            if img is None:
                continue
            yield img
        return

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    idx = 0
    yielded = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % stride == 0:
            yield frame
            yielded += 1
            if max_frames is not None and yielded >= max_frames:
                break
        idx += 1
    cap.release()


def collect_ref_images(ref_args):
    paths = []
    for ref in ref_args:
        if os.path.isdir(ref):
            for ext in IMAGE_EXTS:
                paths.extend(Path(ref).glob(f"*{ext}"))
                paths.extend(Path(ref).glob(f"*{ext.upper()}"))
        else:
            paths.append(Path(ref))
    return [str(p) for p in sorted(set(paths))]


def l2_norm_np(arr):
    if arr.ndim == 1:
        norm = np.linalg.norm(arr, ord=2)
        return arr / norm
    norm = np.linalg.norm(arr, ord=2, axis=1, keepdims=True)
    return arr / norm


class InsightFaceEmbedder:
    def __init__(self, model_name, model_root, device, det_size):
        try:
            from insightface.app import FaceAnalysis
        except Exception as exc:
            raise RuntimeError(
                "insightface is required for identity consistency. "
                "Install it via pip install insightface"
            ) from exc

        providers = ["CPUExecutionProvider"]
        ctx_id = -1
        if device == "gpu":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            ctx_id = 0

        self.app = FaceAnalysis(name=model_name, root=model_root, providers=providers)
        self.app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))

    def embed(self, image_bgr):
        faces = self.app.get(image_bgr)
        if not faces:
            return None
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        emb = face.embedding.astype(np.float32)
        return l2_norm_np(emb)


class CurricularFaceEmbedder:
    def __init__(
        self,
        repo_root,
        backbone_name,
        ckpt_path,
        device,
        det_model_name,
        det_model_root,
        det_size,
    ):
        if not ckpt_path:
            raise RuntimeError("--curricularface-ckpt is required for CurricularFace backend.")
        try:
            import torch
        except Exception as exc:
            raise RuntimeError(
                "torch is required for CurricularFace backend. Install it first."
            ) from exc
        try:
            from insightface.app import FaceAnalysis
            from insightface.utils import face_align
        except Exception as exc:
            raise RuntimeError(
                "insightface is required for CurricularFace backend (face detect/align)."
            ) from exc

        repo_root = Path(repo_root)
        if not repo_root.exists():
            raise RuntimeError(f"CurricularFace repo not found at {repo_root}")
        import sys

        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        try:
            from backbone.model_resnet import ResNet_50, ResNet_101, ResNet_152
            from backbone.model_irse import (
                IR_50,
                IR_101,
                IR_152,
                IR_SE_50,
                IR_SE_101,
                IR_SE_152,
            )
            from backbone.model_mobilefacenet import MobileFaceNet
        except Exception as exc:
            raise RuntimeError("Failed to import CurricularFace backbones.") from exc

        backbone_dict = {
            "ResNet_50": ResNet_50,
            "ResNet_101": ResNet_101,
            "ResNet_152": ResNet_152,
            "IR_50": IR_50,
            "IR_101": IR_101,
            "IR_152": IR_152,
            "IR_SE_50": IR_SE_50,
            "IR_SE_101": IR_SE_101,
            "IR_SE_152": IR_SE_152,
            "MobileFaceNet": MobileFaceNet,
        }
        if backbone_name not in backbone_dict:
            raise RuntimeError(f"Unsupported CurricularFace backbone: {backbone_name}")

        self.torch = torch
        self.device = torch.device("cuda:0" if device == "gpu" and torch.cuda.is_available() else "cpu")
        self.face_align = face_align

        providers = ["CPUExecutionProvider"]
        ctx_id = -1
        if device == "gpu":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            ctx_id = 0
        self.detector = FaceAnalysis(
            name=det_model_name,
            root=det_model_root,
            providers=providers,
        )
        self.detector.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))

        self.backbone = backbone_dict[backbone_name]([112, 112])
        state = torch.load(ckpt_path, map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        cleaned = {}
        for k, v in state.items():
            key = k[7:] if k.startswith("module.") else k
            cleaned[key] = v
        self.backbone.load_state_dict(cleaned, strict=False)
        self.backbone.eval().to(self.device)

    def _preprocess(self, image_bgr):
        img = cv2.resize(image_bgr, (112, 112))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = self.torch.from_numpy(img).float() / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        tensor = (tensor - 0.5) / 0.5
        return tensor

    def _l2_norm_torch(self, input_tensor, axis=1):
        norm = self.torch.norm(input_tensor, 2, axis, True)
        return self.torch.div(input_tensor, norm)

    def embed(self, image_bgr):
        faces = self.detector.get(image_bgr)
        if not faces:
            return None
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        if getattr(face, "kps", None) is not None:
            crop = self.face_align.norm_crop(image_bgr, face.kps, image_size=112)
        else:
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            crop = image_bgr[max(y1, 0) : max(y2, 0), max(x1, 0) : max(x2, 0)]
        if crop is None or crop.size == 0:
            return None
        tensor = self._preprocess(crop).to(self.device)
        with self.torch.no_grad():
            feat = self.backbone(tensor)[0]
            feat = self._l2_norm_torch(feat)
        return feat.squeeze(0).cpu().numpy().astype(np.float32)


class PoseEstimator:
    def __init__(
        self,
        det_size=640,
        device="cpu",
        use_insightface_fallback=True,
        det_model_name="buffalo_l",
        det_model_root="./models/insightface",
        use_prev=True,
    ):
        import mediapipe as mp

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # 3D model points (6-point) for MediaPipe landmarks.
        self.model_points_mp = np.array(
            [
                (0.0, 0.0, 0.0),
                (0.0, -63.6, -12.5),
                (-43.3, 32.7, -26.0),
                (43.3, 32.7, -26.0),
                (-28.9, -28.9, -24.1),
                (28.9, -28.9, -24.1),
            ],
            dtype="double",
        )
        # Landmark indices: nose tip, chin, left eye, right eye, left mouth, right mouth.
        self.landmark_ids = [1, 152, 33, 263, 61, 291]

        # 3D model points (5-point) for InsightFace keypoints:
        # left eye, right eye, nose, left mouth, right mouth.
        self.model_points_5 = np.array(
            [
                (-30.0, 30.0, -30.0),
                (30.0, 30.0, -30.0),
                (0.0, 0.0, 0.0),
                (-25.0, -30.0, -30.0),
                (25.0, -30.0, -30.0),
            ],
            dtype="double",
        )

        self.use_insightface_fallback = use_insightface_fallback
        self.detector = None
        self.use_prev = use_prev
        self.prev_rvec = None
        self.prev_tvec = None
        self.prev_euler = None
        if use_insightface_fallback:
            try:
                from insightface.app import FaceAnalysis
            except Exception:
                self.use_insightface_fallback = False
            else:
                providers = ["CPUExecutionProvider"]
                ctx_id = -1
                if device == "gpu":
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    ctx_id = 0
                self.detector = FaceAnalysis(
                    name=det_model_name,
                    root=det_model_root,
                    providers=providers,
                )
                self.detector.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))

    def _unwrap_euler(self, euler):
        if self.prev_euler is None:
            return euler
        unwrapped = euler.copy()
        for i in range(3):
            while unwrapped[i] - self.prev_euler[i] > 180.0:
                unwrapped[i] -= 360.0
            while unwrapped[i] - self.prev_euler[i] < -180.0:
                unwrapped[i] += 360.0
        return unwrapped

    def _solve_pose(self, image_points, model_points, w, h):
        focal_length = w
        center = (w / 2.0, h / 2.0)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
            dtype="double",
        )
        dist_coeffs = np.zeros((4, 1))

        flag = cv2.SOLVEPNP_ITERATIVE
        if len(model_points) < 6:
            flag = cv2.SOLVEPNP_EPNP

        if self.use_prev and self.prev_rvec is not None and self.prev_tvec is not None:
            success, rotation_vec, translation_vec = cv2.solvePnP(
                model_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                self.prev_rvec,
                self.prev_tvec,
                True,
                flags=flag,
            )
        else:
            success, rotation_vec, translation_vec = cv2.solvePnP(
                model_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=flag,
            )
        if not success:
            return None

        rot_mat, _ = cv2.Rodrigues(rotation_vec)
        proj_mat = np.hstack((rot_mat, translation_vec))
        _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(proj_mat)
        pitch, yaw, roll = [float(a) for a in euler]
        euler = np.array([pitch, yaw, roll], dtype=np.float32)
        euler = self._unwrap_euler(euler)

        self.prev_rvec = rotation_vec
        self.prev_tvec = translation_vec
        self.prev_euler = euler
        return euler, rotation_vec, translation_vec

    def _estimate_mediapipe(self, image_bgr):
        h, w = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        if not results.multi_face_landmarks:
            return None
        landmarks = results.multi_face_landmarks[0].landmark

        image_points = []
        for idx in self.landmark_ids:
            lm = landmarks[idx]
            image_points.append((lm.x * w, lm.y * h))
        image_points = np.array(image_points, dtype="double")
        return self._solve_pose(image_points, self.model_points_mp, w, h)

    def _estimate_insightface(self, image_bgr):
        if not self.detector:
            return None
        h, w = image_bgr.shape[:2]
        faces = self.detector.get(image_bgr)
        if not faces:
            return None
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        kps = getattr(face, "kps", None)
        if kps is None or len(kps) != 5:
            return None
        image_points = np.array(kps, dtype="double")
        return self._solve_pose(image_points, self.model_points_5, w, h)

    def estimate_rt(self, image_bgr):
        result = self._estimate_mediapipe(image_bgr)
        if result is not None:
            return result
        if self.use_insightface_fallback:
            return self._estimate_insightface(image_bgr)
        return None

    def estimate(self, image_bgr):
        result = self.estimate_rt(image_bgr)
        if result is None:
            return None
        euler, _, _ = result
        return euler


def cosine_similarity(a, b):
    a = l2_norm_np(a)
    b = l2_norm_np(b)
    return float(np.dot(a, b))


def compute_identity_consistency(video_path, ref_paths, embedder, stride, max_frames, missing_policy):
    ref_embs = []
    for ref_path in ref_paths:
        img = cv2.imread(ref_path)
        if img is None:
            continue
        emb = embedder.embed(img)
        if emb is not None:
            ref_embs.append(emb)
    if not ref_embs:
        raise RuntimeError("No valid face found in reference images.")

    sims = []
    total = 0
    missing = 0
    for frame in iter_frames(video_path, stride=stride, max_frames=max_frames):
        total += 1
        emb = embedder.embed(frame)
        if emb is None:
            missing += 1
            if missing_policy == "fail":
                raise RuntimeError("Face not found in video frame.")
            if missing_policy == "zero":
                sims.append(0.0)
            continue
        sims.append(max(cosine_similarity(emb, r) for r in ref_embs))

    result = None
    if sims:
        result = float(np.mean(sims))

    return result, {"total_frames": total, "used_frames": len(sims), "missing_frames": missing}


def compute_motion_dynamic(
    gen_path,
    cond_path,
    pose_estimator,
    stride,
    max_frames,
    missing_policy,
    return_angles=False,
):
    diffs = []
    total = 0
    missing = 0
    angle_records = [] if return_angles else None

    gen_iter = iter_frames(gen_path, stride=stride, max_frames=max_frames)
    cond_iter = iter_frames(cond_path, stride=stride, max_frames=max_frames)

    if isinstance(pose_estimator, (tuple, list)) and len(pose_estimator) == 2:
        gen_estimator, cond_estimator = pose_estimator
    else:
        gen_estimator = cond_estimator = pose_estimator

    for idx, (gen_frame, cond_frame) in enumerate(zip(gen_iter, cond_iter)):
        total += 1
        gen_pose = gen_estimator.estimate(gen_frame)
        cond_pose = cond_estimator.estimate(cond_frame)
        missing_flag = gen_pose is None or cond_pose is None
        diff_value = None
        if missing_flag:
            missing += 1
            if missing_policy == "fail":
                raise RuntimeError("Face pose not found in paired frames.")
            if missing_policy == "zero":
                diff_value = 0.0
                diffs.append(diff_value)
        else:
            diff_value = float(np.linalg.norm(gen_pose - cond_pose))
            diffs.append(diff_value)

        if return_angles:
            angle_records.append(
                {
                    "idx": idx,
                    "gen": gen_pose.tolist() if gen_pose is not None else None,
                    "cond": cond_pose.tolist() if cond_pose is not None else None,
                    "diff": diff_value,
                    "missing": missing_flag,
                }
            )

    result = None
    if diffs:
        result = float(np.mean(diffs))

    stats = {"total_pairs": total, "used_pairs": len(diffs), "missing_pairs": missing}
    if return_angles:
        return result, stats, angle_records
    return result, stats


def parse_args():
    parser = argparse.ArgumentParser(description="Compute IC2 and MD metrics.")
    parser.add_argument("--gen", required=True, help="Generated video path or frame dir")
    parser.add_argument("--cond", help="Condition video path or frame dir (for MD)")
    parser.add_argument("--refs", nargs="+", help="Reference image paths or dirs (for IC2)")
    parser.add_argument("--skip-ic", action="store_true", help="Skip identity consistency")
    parser.add_argument("--skip-md", action="store_true", help="Skip motion dynamic")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to evaluate")
    parser.add_argument(
        "--missing-policy",
        choices=["skip", "zero", "fail"],
        default="skip",
        help="Policy when face/pose is missing",
    )
    parser.add_argument(
        "--face-backend",
        choices=["insightface", "curricularface"],
        default="insightface",
        help="Face embedding backend",
    )
    parser.add_argument("--face-model-name", default="buffalo_l", help="InsightFace model name")
    parser.add_argument("--face-model-root", default="./models/insightface", help="Model root dir")
    parser.add_argument("--det-size", type=int, default=640, help="Detector input size")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu", help="Device for insightface")
    parser.add_argument(
        "--curricularface-repo",
        default="./CurricularFace",
        help="Local path to CurricularFace repo",
    )
    parser.add_argument(
        "--curricularface-backbone",
        default="IR_101",
        help="CurricularFace backbone name",
    )
    parser.add_argument(
        "--curricularface-ckpt",
        default=None,
        help="Path to CurricularFace backbone checkpoint (.pth)",
    )
    parser.add_argument(
        "--dump-angles",
        default=None,
        help="Save per-frame pose angles to JSON (for MD)",
    )
    parser.add_argument("--out", help="Output JSON path")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.skip_ic and not args.refs:
        raise SystemExit("--refs is required unless --skip-ic is set")
    if not args.skip_md and not args.cond:
        raise SystemExit("--cond is required unless --skip-md is set")

    results = {}

    if not args.skip_ic:
        ref_paths = collect_ref_images(args.refs)
        if args.face_backend == "curricularface":
            embedder = CurricularFaceEmbedder(
                repo_root=args.curricularface_repo,
                backbone_name=args.curricularface_backbone,
                ckpt_path=args.curricularface_ckpt,
                device=args.device,
                det_model_name=args.face_model_name,
                det_model_root=args.face_model_root,
                det_size=args.det_size,
            )
        else:
            embedder = InsightFaceEmbedder(
                model_name=args.face_model_name,
                model_root=args.face_model_root,
                device=args.device,
                det_size=args.det_size,
            )
        ic2, ic_stats = compute_identity_consistency(
            args.gen,
            ref_paths,
            embedder,
            stride=args.stride,
            max_frames=args.max_frames,
            missing_policy=args.missing_policy,
        )
        results["identity_consistency"] = {
            "ic2": ic2,
            "backend": args.face_backend,
            **ic_stats,
        }

    if not args.skip_md:
        pose_estimator = (PoseEstimator(), PoseEstimator())
        md_result = compute_motion_dynamic(
            args.gen,
            args.cond,
            pose_estimator,
            stride=args.stride,
            max_frames=args.max_frames,
            missing_policy=args.missing_policy,
            return_angles=bool(args.dump_angles),
        )
        if args.dump_angles:
            md, md_stats, angle_records = md_result
            angle_payload = {
                "gen": args.gen,
                "cond": args.cond,
                "stride": args.stride,
                "max_frames": args.max_frames,
                "missing_policy": args.missing_policy,
                "frames": angle_records,
            }
            Path(args.dump_angles).parent.mkdir(parents=True, exist_ok=True)
            Path(args.dump_angles).write_text(json.dumps(angle_payload, indent=2))
        else:
            md, md_stats = md_result
        results["motion_dynamic"] = {"md": md, **md_stats}

    output = json.dumps(results, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(output)
    print(output)


if __name__ == "__main__":
    main()
