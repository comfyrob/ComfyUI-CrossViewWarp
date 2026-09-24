"""Two helper nodes for running CrossView Warp as a hosted, two-stage web demo.

The upstream pack is built for the ComfyUI canvas: the live preview caches the
clip in the server process and a custom route re-warps it on every drag. A
managed Comfy API deployment offers neither - jobs are stateless and only the
v2 job/asset endpoints are reachable - so the demo splits the run in two:

  stage 1  CrossViewPrepareClip -> Run MoGe Inference -> CrossViewGeometryExport
           returns one binary file; the browser re-warps it locally while the
           visitor aims the camera
  stage 2  CrossViewPrepareClip -> Run MoGe Inference -> CrossView Warp -> H3
           the camera the visitor chose goes into the warp node's widgets

Both stages start with the same CrossViewPrepareClip call, so the frames the
browser previews are the frames the generation warps.
"""

import io
import json
import math
import os
import struct
import zlib

import numpy as np
import torch

import comfy.utils
import folder_paths

from .crossview_preview_node import _fit

FPS = 24.0

# name -> (w, h); "source" keeps the clip's own shape.
ASPECTS = {
    "source": None,
    "16:9": (16, 9),
    "9:16": (9, 16),
    "1:1": (1, 1),
    "4:3": (4, 3),
    "3:4": (3, 4),
    "21:9": (21, 9),
}

MAGIC = b"CVGEO1\0\0"


def _valid_length(n):
    """MiniMax H3 takes 17k + 5 frames; round UP like the source workflow did."""
    n = max(5, int(n))
    return n + (5 - n % 17) % 17


def _fit_length(n, available):
    """Largest valid length that is <= both the request and the clip."""
    n = min(_valid_length(n), available)
    return n - (n - 5) % 17 if n >= 5 else 0


def _target_size(aspect, src_w, src_h, megapixels, multiple):
    """ResolutionSelector's arithmetic, with the clip's own ratio for "source"."""
    wr, hr = ASPECTS[aspect] or (src_w, src_h)
    scale = math.sqrt(megapixels * 1024 * 1024 / (wr * hr))
    w = max(multiple, round(wr * scale / multiple) * multiple)
    h = max(multiple, round(hr * scale / multiple) * multiple)
    return int(w), int(h)


def _trim_audio(audio, seconds):
    """The first `seconds` of the clip's audio, so it lines up with the frames
    kept from its start; silence of that length when the clip is silent."""
    if audio is None or audio.get("waveform") is None:
        rate = 44100
        return {"waveform": torch.zeros(1, 2, int(round(seconds * rate))), "sample_rate": rate}
    rate = int(audio["sample_rate"])
    wave = audio["waveform"][..., : int(round(seconds * rate))]
    return {"waveform": wave.contiguous(), "sample_rate": rate}


class CrossViewPrepareClip:
    """Video -> the exact frames both stages warp and generate from.

    Resamples to 24 fps (the source workflow assumed 24 without checking),
    trims to the requested duration on H3's 17k+5 grid, then centre-crops and
    resizes to the chosen aspect at the requested pixel budget.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "duration": ("FLOAT", {"default": 5.0, "min": 0.2, "max": 15.0, "step": 0.1,
                    "tooltip": "Seconds of the clip to use, from its start. Snapped to the "
                    "nearest frame count H3 accepts (17k+5 at 24 fps) and capped at the clip."}),
                "aspect_ratio": (list(ASPECTS), {"default": "source",
                    "tooltip": "Output shape. 'source' keeps the clip's own ratio; the others "
                    "centre-crop to it."}),
                "megapixels": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 4.0, "step": 0.05}),
                "multiple": ("INT", {"default": 32, "min": 8, "max": 128, "step": 8}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "AUDIO")
    RETURN_NAMES = ("frames", "width", "height", "length", "audio")
    OUTPUT_TOOLTIPS = (
        "The frames both stages use, at 24 fps.",
        "Frame width.",
        "Frame height.",
        "Frame count, on H3's 17k+5 grid.",
        "The clip's own audio over exactly those frames, silence if it has none.",
    )
    FUNCTION = "prepare"
    CATEGORY = "CrossView"

    def prepare(self, video, duration, aspect_ratio, megapixels, multiple):
        comps = video.get_components()
        images = comps.images                                    # [B,H,W,C]
        src_fps = float(comps.frame_rate) or FPS
        count, src_h, src_w = images.shape[0], images.shape[1], images.shape[2]

        available = int(math.floor((count - 1) * FPS / src_fps)) + 1
        length = _fit_length(round(duration * FPS), available)
        if length < 5:
            raise ValueError(f"CrossViewPrepareClip: the clip holds {available} frames at "
                             f"24 fps; H3 needs at least 5.")
        idx = [min(count - 1, round(i * src_fps / FPS)) for i in range(length)]
        frames = images[idx]

        width, height = _target_size(aspect_ratio, src_w, src_h, megapixels, multiple)
        x = frames.movedim(-1, 1)                                  # [B,C,H,W]
        method = "area" if width * height < src_w * src_h else "bicubic"
        x = comfy.utils.common_upscale(x, width, height, method, "center")
        frames = x.movedim(1, -1).clamp(0.0, 1.0).contiguous()
        return (frames, width, height, length, _trim_audio(comps.audio, length / FPS))


class CrossViewGeometryExport:
    """Frames + MoGe geometry -> one file the browser can re-warp on its own.

    Holds what the upstream live preview caches in server memory (see
    crossview_preview_node.seed_cache), at the same preview scale, so a
    browser port of _warp_frame reproduces the node's own preview.

    Layout: MAGIC, uint32 LE header length, UTF-8 JSON header, then the blobs
    the header points at (offsets from the start of the blob area):
      frames  one JPEG per frame
      depth   float16 metres, [n,h,w], NaN = no geometry; byte-shuffled
              (every low byte, then every high byte) and zlib-compressed.
              float16 steps are ~0.1% of depth, far below a pixel of warp.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "moge_geometry": ("MOGE_GEOMETRY",),
                "preview_size": ("INT", {"default": 384, "min": 128, "max": 1024, "step": 32,
                    "tooltip": "Longest side of the exported preview. Warp geometry follows the "
                    "frame width, so a smaller preview is the same warp at lower resolution."}),
                "jpeg_quality": ("INT", {"default": 88, "min": 50, "max": 100}),
                "filename_prefix": ("STRING", {"default": "crossview/geometry"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "export"
    OUTPUT_NODE = True
    CATEGORY = "CrossView"

    def export(self, frames, moge_geometry, preview_size, jpeg_quality, filename_prefix):
        from PIL import Image

        size = int(preview_size)
        small = _fit(frames.detach().float().cpu(), size)
        rgb = (small.clamp(0, 1).numpy() * 255.0).astype(np.uint8)
        n, h, w = rgb.shape[:3]

        # Exactly seed_cache's metric path: bilinear depth, nearest mask, then
        # build()'s own masking of invalid and non-positive depth.
        md = moge_geometry["depth"]
        z = _fit(md[..., None].detach().float().cpu(), size).squeeze(-1).numpy()
        mm = moge_geometry.get("mask")
        if mm is not None:
            m = _fit(mm[..., None].detach().float().cpu(), size, "nearest").squeeze(-1).numpy()
            z = np.where(m > 0.5, z, np.nan)
        z = np.where(np.isfinite(z) & (z > 0), z, np.nan).astype(np.float16)

        K = moge_geometry.get("intrinsics")
        fx_norm = float(K[0][0, 0]) if K is not None else None

        blobs = io.BytesIO()
        jpegs = []
        for i in range(n):
            buf = io.BytesIO()
            Image.fromarray(rgb[i]).save(buf, format="JPEG", quality=int(jpeg_quality))
            jpegs.append({"offset": blobs.tell(), "length": buf.tell()})
            blobs.write(buf.getvalue())
        planes = np.ascontiguousarray(z).astype("<f2").view(np.uint8).reshape(-1, 2).T
        depth_raw = zlib.compress(np.ascontiguousarray(planes).tobytes(), 6)
        depth = {"offset": blobs.tell(), "length": len(depth_raw), "dtype": "float16",
                 "shape": [n, h, w], "encoding": "shuffle2+zlib"}
        blobs.write(depth_raw)

        header = {
            "version": 1,
            "frames": n,
            "width": w,
            "height": h,
            "fps": FPS,
            "source_width": int(frames.shape[2]),
            "source_height": int(frames.shape[1]),
            "fx_norm": fx_norm,
            "jpegs": jpegs,
            "depth": depth,
        }
        head = json.dumps(header, separators=(",", ":")).encode("utf-8")

        out_dir = folder_paths.get_output_directory()
        full_dir, name, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, out_dir, w, h)
        filename = f"{name}_{counter:05}_.cvgeo"
        with open(os.path.join(full_dir, filename), "wb") as f:
            f.write(MAGIC)
            f.write(struct.pack("<I", len(head)))
            f.write(head)
            f.write(blobs.getvalue())

        # "files" is what the v2 API reports as an output of kind "file".
        return {"ui": {"files": [{"filename": filename, "subfolder": subfolder,
                                  "type": "output"}]}}


NODE_CLASS_MAPPINGS = {
    "CrossViewPrepareClip": CrossViewPrepareClip,
    "CrossViewGeometryExport": CrossViewGeometryExport,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CrossViewPrepareClip": "CrossView Prepare Clip",
    "CrossViewGeometryExport": "CrossView Geometry Export",
}
