"""1 단계의 크기 바꾸기 증강 (train_rel.py --jitter, 팔 rtj) 을 끝까지 — 아주 작은 가짜 영상 (126×154) 으로, 따로 띄운 프로세스에서.
부품 시험 (test_parts.py) 에 넣지 않는 까닭: 그쪽은 이미 3 GB 한도 가까이 쓴다 (2026-11-24 에 256×640 을 1.3 배로 키운 역전파를 넣었다가 한도에 걸렸다).
여기는 1.3 배 (배치를 둘로 나눠 계산하는 길) 까지 돌아도 수백 MB 다.

  TD_CODE=<Depth Anything V2 코드> python -I -B tests/test_rel_jitter.py
"""
import json
import os
import random
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402


def tiny_tree(root, rng, h=126, w=154):
    """MS2 꼴의 작은 묶음: train 두 시퀀스 · val 한 시퀀스에 3 장씩."""
    for name, k in (("train", 2), ("val", 1)):
        seqs = [s.strip() for s in open(os.path.join(HERE, "..", "splits", "ms2", f"{name}_list.txt")) if s.strip()]
        for s in seqs[:k]:
            img, dep = os.path.join(root, "sync_data", s, "thr", "img_left"), os.path.join(root, "proj_depth", s, "thr", "depth_filtered")
            os.makedirs(img)
            os.makedirs(dep)
            for i in range(3):
                Image.fromarray(rng.integers(3000, 6000, (h, w), dtype=np.uint16)).save(os.path.join(img, f"{i:06d}.png"))
                d = (rng.uniform(2, 60, (h, w)) * 256).astype(np.uint16)
                d[rng.random((h, w)) < 0.5] = 0
                Image.fromarray(d).save(os.path.join(dep, f"{i:06d}.png"))


class RelJitter(unittest.TestCase):
    def test_train_rel_jitter(self):
        with tempfile.TemporaryDirectory() as t:
            tiny_tree(os.path.join(t, "data"), np.random.default_rng(0))
            os.environ["TD_DATA"] = os.path.join(t, "data")
            import train_rel
            scales = (0.71, 1.3)
            self.assertEqual({random.Random((0 * 1000003 + st) * 4 + 3).choice(scales) for st in range(3)}, set(scales))   # 세 스텝에 두 배율이 모두 뽑힌다
            argv = sys.argv
            sys.argv = ["train_rel", "--encoder", "vits", "--epochs", "1", "--bs", "2", "--workers", "0", "--amp", "0", "--device", "cpu", "--jitter", "0.71,1.3",
                        "--out", os.path.join(t, "rtj_s0", "rel"), "--ckpt", os.path.join(t, "ck")]
            try:
                self.assertEqual(train_rel.main(), 0)
            finally:
                sys.argv = argv
            done = json.load(open(os.path.join(t, "rtj_s0", "rel", "done.json")))
            self.assertEqual((done["steps"], done["jitter"]), (3, [0.71, 1.3]))               # 6 장 / 배치 2
            self.assertTrue(os.path.exists(os.path.join(t, "ck", "rel.pt")))


if __name__ == "__main__":
    unittest.main(verbosity=1)
