"""Smoke test for the demo nodes. Run with the ComfyUI venv:

    PYTHONPATH=<ComfyUI code dir> <venv python> tests/test_demo_nodes.py
"""
import importlib.util
import json
import os
import struct
import sys
import tempfile
import types
import zlib
from fractions import Fraction

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("crossviewwarp", os.path.join(HERE, "__init__.py"),
                                              submodule_search_locations=[HERE])
pack = importlib.util.module_from_spec(spec)
sys.modules["crossviewwarp"] = pack
spec.loader.exec_module(pack)
demo = sys.modules["crossviewwarp.crossview_demo_nodes"]
import folder_paths  # noqa: E402


class FakeVideo:
    def __init__(self, n, h, w, fps):
        self.images = torch.rand(n, h, w, 3)
        self.fps = Fraction(fps)

    def get_components(self):
        return types.SimpleNamespace(images=self.images, frame_rate=self.fps, audio=None)


def test_lengths():
    assert demo._valid_length(120) == 124          # 5 s, as the source workflow computed
    assert demo._valid_length(1) == 5
    assert demo._fit_length(124, 48) == 39          # capped to the clip, still 17k+5
    assert demo._fit_length(124, 500) == 124
    assert demo._target_size("16:9", 1, 1, 0.5, 32) == (960, 544)
    assert demo._target_size("source", 640, 360, 0.5, 32) == (960, 544)
    assert demo._target_size("9:16", 1, 1, 0.5, 32) == (544, 960)


def test_prepare_resamples_and_trims():
    node = demo.CrossViewPrepareClip()
    # 60 frames at 30 fps = 2 s -> 48 frames at 24 fps -> 39 valid
    frames, w, h, length = node.prepare(FakeVideo(60, 360, 640, 30), 5.0, "source", 0.5, 32)
    assert (w, h, length) == (960, 544, 39) and tuple(frames.shape) == (39, 544, 960, 3)
    frames, w, h, length = node.prepare(FakeVideo(200, 360, 640, 24), 1.0, "1:1", 0.5, 32)
    assert length == 39 and w == h and tuple(frames.shape) == (39, h, w, 3)  # 24 -> next 17k+5


def read_cvgeo(path):
    raw = open(path, "rb").read()
    assert raw[:8] == demo.MAGIC
    (hl,) = struct.unpack("<I", raw[8:12])
    head = json.loads(raw[12:12 + hl])
    blob = raw[12 + hl:]
    d = head["depth"]
    planes = np.frombuffer(zlib.decompress(blob[d["offset"]:d["offset"] + d["length"]]), np.uint8)
    z = planes.reshape(2, -1).T.copy().view("<f2").reshape(d["shape"])
    return head, blob, z


def test_export_roundtrip():
    out = tempfile.mkdtemp()
    folder_paths.set_output_directory(out)
    n, H, W = 5, 544, 960
    frames = torch.rand(n, H, W, 3)
    depth = torch.rand(n, H, W) * 10 + 1
    mask = torch.ones(n, H, W, dtype=torch.bool)
    mask[:, :40] = False
    K = torch.tensor([[[0.9, 0, 0.5], [0, 1.6, 0.5], [0, 0, 1]]]).repeat(n, 1, 1)
    res = demo.CrossViewGeometryExport().export(
        frames, {"depth": depth, "mask": mask, "intrinsics": K}, 384, 88, "crossview/geometry")
    f = res["ui"]["files"][0]
    path = os.path.join(out, f["subfolder"], f["filename"])
    head, blob, z = read_cvgeo(path)
    assert (head["frames"], head["width"], head["height"]) == (n, 384, 218)
    assert abs(head["fx_norm"] - 0.9) < 1e-6
    assert z.shape == (n, 218, 384)
    assert np.isnan(z[:, 0]).all() and np.isfinite(z[:, -1]).all()
    fin = z[np.isfinite(z)]
    assert fin.min() >= 1 and fin.max() <= 11
    j = head["jpegs"][0]
    assert blob[j["offset"]:j["offset"] + 2] == b"\xff\xd8"
    print(f"cvgeo: {os.path.getsize(path) / 1e6:.2f} MB for {n} frames")


if __name__ == "__main__":
    test_lengths()
    test_prepare_resamples_and_trims()
    test_export_roundtrip()
    print("ok")
