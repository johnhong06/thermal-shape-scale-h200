"""모양 실험 (run.sh shape) 의 1 단계 변형을 끝까지 — 가짜 묶음 · 작은 인코더 (ViT-S, 무작위 초기화) 로, 따로 띄운 프로세스에서.
부품 시험 (test_parts.py) 에 넣지 않는 까닭: 그쪽은 이미 2.3 GB 쯤 쓰고, 여기에 512×640 쌍으로 일곱 변형 (원래 모델 사본 · 기울기 재기 포함) 을 더하면
3 GB 한도에 걸린다 (2026-11-26 에 두 번 걸렸다 — 따로 띄워도 걸려서 쌍 영상 · 라벨을 반으로 줄였다. 크기는 학습 · 평가 코드가 영상에서 읽으므로 흐름은 같다).
손실 · 토큰 · 경계 F1 의 부품은 test_parts.py 의 test_shape_losses 에 있다.

  TD_CODE=<Depth Anything V2 코드> python -I -B tests/test_shape.py
"""
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
import gc  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import data as D  # noqa: E402
from test_parts import fake_tree  # noqa: E402


def shrink_pairs(root):
    """쌍 (열화상 · RGB · 라벨 두 가지) 을 가로세로 반으로 — 같은 쌍은 같은 크기를 유지한다."""
    for name in D.LABEL_SETS:
        for key, tp, rp in D.pair_frames(name, root):
            for f in (tp, rp, D.label_path(name, key, root), D.label_path(name, key, root, "labels_cam")):
                if os.path.exists(f):
                    im = Image.open(f)
                    im.resize((im.width // 2, im.height // 2), Image.NEAREST).save(f)


class Shape(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        fake_tree(cls.tmp.name, np.random.default_rng(0))
        shrink_pairs(cls.tmp.name)
        os.environ["TD_DATA"] = cls.tmp.name

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_shape_train(self):
        """모양 실험의 1 단계 변형을 끝까지 (가짜 묶음, 작은 인코더) → 모양 평가 → 세 표."""
        import rel_eval
        import train_rel
        import model as M

        def run(mod, args):
            argv = sys.argv
            sys.argv = [mod.__name__] + args
            try:
                return mod.main()
            finally:
                sys.argv = argv
        with tempfile.TemporaryDirectory() as t:
            init = os.path.join(t, "init.pth")                                                                     # '학습을 시작한 가중치' 노릇 (H200 에서는 공식 체크포인트)
            torch.manual_seed(0)
            torch.save(M.build_rel("vits").state_dict(), init)
            base = ["--encoder", "vits", "--weights", init, "--epochs", "1", "--bs", "2", "--workers", "0", "--amp", "0", "--device", "cpu", "--diag_every", "1"]
            for v, fl in (("pair", ["--pairs"]), ("edge", ["--pairs", "--grad_match", "2.0"]), ("freeze", ["--pairs", "--freeze_enc"]),
                          ("thr_cos", ["--pairs", "--fa", "thr_cos"]), ("thr_gram", ["--pairs", "--fa", "thr_gram"]),
                          ("rgb_cos", ["--pairs", "--fa", "rgb_cos"]), ("rgb_gram", ["--pairs", "--fa", "rgb_gram"])):
                out, ck = os.path.join(t, "runs", f"shape_{v}_s0", "rel"), os.path.join(t, "ck", v)
                self.assertEqual(run(train_rel, base + ["--out", out, "--ckpt", ck, "--pair_bs", "1", "--hold", "freiburg"] + fl), 0)
                gc.collect()
                done = json.load(open(os.path.join(out, "done.json")))
                self.assertEqual((done["hold"], done["pairs"], done["steps"]), (["freiburg"], [4], 3))            # Freiburg 는 학습에서 빠진다
                fa = v if v.startswith(("thr_", "rgb_")) else None
                self.assertEqual((done["fa"], done["grad_match"], done["freeze_enc"]), (fa, 2.0 if v == "edge" else None, v == "freeze"))
                le = done["last_epoch"]
                if fa:
                    form = v.split("_")[1]
                    self.assertEqual((done["fa_weight"], done["fa_margin"]), (train_rel.FA_WEIGHT[form], 0.85 if form == "cos" else None))
                    self.assertTrue(np.isfinite(le["fa"]) and -1.0 <= le["fa_sim"] <= 1.0 and 0.0 <= le["fa_on"] <= 1.0)
                    if v.startswith("rgb_"):                                                                        # 짝 RGB 목표는 처음부터 손실이 있다 → 기울기 코사인을 잰다
                        self.assertTrue(le["gn"] >= 1 and -1.0 <= le["gcos_ms2"] <= 1.0 and -1.0 <= le["gcos_pair"] <= 1.0 and le["gratio"] > 0)
                    if v == "thr_cos":                                                                              # 같은 열화상 목표는 처음에 원래 모델 그대로라 여유 안 → 손실 0
                        self.assertEqual(le["fa"], 0.0)
                else:
                    self.assertIsNone(le["fa"])
                if v == "freeze":                                                                                   # 인코더는 그대로, 헤드만 바뀐다
                    sd, sd0 = torch.load(os.path.join(ck, "rel.pt"), weights_only=True), torch.load(init, weights_only=True)
                    self.assertTrue(all(torch.equal(sd[k], sd0[k]) for k in sd if "pretrained" in k))
                    self.assertFalse(all(torch.equal(sd[k], sd0[k]) for k in sd if "pretrained" not in k))
            with self.assertRaises(SystemExit):                                                                    # 쌍 없이 정렬 · 빼 두기는 안 된다
                run(train_rel, base + ["--out", os.path.join(t, "x", "rel"), "--ckpt", os.path.join(t, "x"), "--fa", "rgb_cos"])
            for v in ("pair", "rgb_gram"):
                out = os.path.join(t, "runs", f"shape_{v}_s0", "eval")
                self.assertEqual(run(rel_eval, ["--ckpt", os.path.join(t, "ck", v, "rel.pt"), "--out", out, "--encoder", "vits", "--bs", "2", "--workers", "0",
                                                "--device", "cpu", "--hold", "freiburg"] + (["--origin", init] if v == "rgb_gram" else [])), 0)
            r = json.load(open(os.path.join(t, "runs", "shape_rgb_gram_s0", "eval", "rel_metrics.json")))
            self.assertEqual(set(r["sets"]), set(rel_eval.GT_SETS))
            self.assertTrue(all(0 < v["abs_rel"] < 5 and v["n"] > 0 for v in r["sets"].values()))
            h = r["hold"]["freiburg"]
            self.assertTrue(h["n"] == 1 and set(h["repr"]) == {"drift", "cross", "cross_origin", "gram", "gram_origin"} and -1 <= h["repr"]["drift"] <= 1)
            self.assertTrue(h["edge_f1"] is None or 0.0 <= h["edge_f1"] <= 1.0)
            self.assertEqual(rel_eval.variant_of("shape_rgb_gram_s12"), ("rgb_gram", "12"))
            self.assertEqual(run(rel_eval, ["--summary", os.path.join(t, "runs"), "--out", os.path.join(t, "summ")]), 0)
            tab = json.load(open(os.path.join(t, "summ", "shape.json")))
            self.assertEqual(list(tab), ["shape_pair_s0", "shape_rgb_gram_s0"])                                    # 표의 순서 = rel_eval.ORDER
            self.assertTrue(tab["shape_rgb_gram_s0"]["ms2_test"] > 0 and tab["shape_rgb_gram_s0"]["hold_abs_rel"] > 0 and tab["shape_pair_s0"]["repr"] is None)
            self.assertTrue(tab["shape_rgb_gram_s0"]["gcos_pair"] is not None and tab["shape_pair_s0"]["gcos_pair"] is None)


if __name__ == "__main__":
    unittest.main()
