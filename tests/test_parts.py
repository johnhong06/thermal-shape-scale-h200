"""부품 시험 (CPU, 가짜 데이터) — 지표 · 손실 · 톤매핑 · 데이터 순서 · 모델 입출력 (한꺼번에 · 모양 + 스케일) · 평가 · 표 · 묶음 풀기 · 선생 라벨 · 선생 점검의 계산.

  TD_CODE=<공식 코드 폴더> <파이썬> -I -B tests/test_parts.py
TD_CODE 는 setup/fetch_code.sh 가 받은 폴더 (tests/run_local.sh 가 준비한다). 모델은 무작위 초기화한 vits 를 쓴다 (가중치를 받지 않음).
"""
import io
import json
import os
import sys
import tarfile
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "setup"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import common as C  # noqa: E402
import data as D  # noqa: E402


def fake_tree(root, rng):
    """작은 가짜 묶음: MS2 분할마다 첫 시퀀스에 3 장 (train 은 두 시퀀스), FIReStereo 낮 · 밤 폴더 2 장씩."""
    def put(img, dep, n, h, fmt="{:06d}.png"):
        os.makedirs(img, exist_ok=True)
        os.makedirs(dep, exist_ok=True)
        for i in n:
            Image.fromarray(rng.integers(3000, 6000, (h, 640), dtype=np.uint16)).save(os.path.join(img, fmt.format(i)))
            d = (rng.uniform(2, 60, (h, 640)) * 256).astype(np.uint16)
            d[rng.random((h, 640)) < 0.7] = 0                       # 정답이 드문 LiDAR 처럼
            Image.fromarray(d).save(os.path.join(dep, fmt.format(i)))
    for name in D.MS2_SETS:
        seqs = [s.strip() for s in open(os.path.join(HERE, "..", "splits", "ms2", f"{name}_list.txt")) if s.strip()]
        for s in seqs[:2 if name == "train" else 1]:
            put(os.path.join(root, "sync_data", s, "thr", "img_left"), os.path.join(root, "proj_depth", s, "thr", "depth_filtered"), (7, 3, 5), 256)
    for d in ("hawkins_9_test", "frick_9_test"):
        put(os.path.join(root, "firestereo", d, "img_left"), os.path.join(root, "firestereo", d, "depth_filtered"), (2, 1), 512)
    for s in ("outdoor_robust_night1", "outdoor_robust_night2", "outdoor_robust_day1", "indoor_robust_dark", "indoor_aggresive_local"):
        put(os.path.join(root, "vivid", s, "Thermal"), os.path.join(root, "vivid", s, "Depth_T"), (4, 2), 512)
    q = os.path.join(root, "sthereo", "sthereo_09_valley_evening")
    put(os.path.join(q, "Thermal"), os.path.join(q, "Depth_T"), (1630073700000000002, 1630073700000000001), 512, "{:019d}.png")
    rgb = lambda h, w: Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8))  # noqa: E731
    os.makedirs(os.path.join(root, "vivid", "outdoor_robust_day1", "RGB"))
    for i in (4, 2):
        rgb(480, 640).save(os.path.join(root, "vivid", "outdoor_robust_day1", "RGB", f"{i:06d}.png"))
    for d, n in (("pairs/kaist/set00/V000/lwir", "I00010.jpg"), ("pairs/kaist/set00/V000/visible", "I00010.jpg"), ("pairs/kaist/set03/V001/lwir", "I00000.jpg"),
                 ("pairs/kaist/set03/V001/visible", "I00000.jpg"), ("pairs/llvip/infrared", "010001.jpg"), ("pairs/llvip/visible", "010001.jpg")):
        os.makedirs(os.path.join(root, d), exist_ok=True)
        (rgb(512, 640).convert("L") if d.endswith("infrared") else rgb(512, 640)).save(os.path.join(root, d, n))
    for d, n, im in (("pairs/freiburg/seq_00_day/03/ir", "1570722275_4921181000.png", Image.fromarray(rng.integers(20000, 24000, (298, 640), dtype=np.uint16))),
                     ("pairs/freiburg/seq_00_day/03/rgb", "1570722275_4921181000.jpg", rgb(298, 640)),
                     ("pairs/tartan/day1/20250822_115703/thermal", "00000010.png", rgb(512, 640).convert("L")),
                     ("pairs/tartan/day1/20250822_115703/rgb", "00000010.jpg", rgb(512, 640))):
        os.makedirs(os.path.join(root, d), exist_ok=True)
        im.save(os.path.join(root, d, n))
    open(os.path.join(root, "pairs", "freiburg", "NOTICE.txt"), "w").write("notice")     # 묶음에 든 안내 파일 — 사진 목록에 섞이면 안 된다
    for name in D.LABEL_SETS:                                      # 쌍의 선생 라벨 (src/labels.py 가 쓰는 꼴): 깊이 × 256, 못 쓰는 화소는 0
        for key, tp, _ in D.pair_frames(name, root):
            h, w = Image.open(tp).size[::-1]
            lab = (rng.uniform(3, 40, (h, w)) * 256).astype(np.uint16)
            lab[: h // 8] = 0
            os.makedirs(os.path.dirname(D.label_path(name, key, root)), exist_ok=True)
            Image.fromarray(lab).save(D.label_path(name, key, root))


class Parts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        fake_tree(cls.tmp.name, np.random.default_rng(0))
        os.environ["TD_DATA"] = cls.tmp.name
        torch.manual_seed(0)
        import model as M
        cls.M = M
        cls.model = M.build("vits").eval()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_metrics(self):
        g = torch.rand(5000, dtype=torch.float64) * 70 + 1
        p = g * (1 + 0.2 * torch.randn(5000, dtype=torch.float64)).clamp(0.3, 3)
        m = C.depth_metrics(p, g)
        gn, pn = g.numpy(), p.numpy()
        self.assertAlmostEqual(m["abs_rel"], float(np.mean(np.abs(pn - gn) / gn)), places=10)
        self.assertAlmostEqual(m["rmse"], float(np.sqrt(np.mean((pn - gn) ** 2))), places=10)
        self.assertAlmostEqual(m["rmse_log"], float(np.sqrt(np.mean((np.log(pn) - np.log(gn)) ** 2))), places=10)
        self.assertAlmostEqual(m["d1"], float(np.mean(np.maximum(pn / gn, gn / pn) < 1.25)), places=10)

    def test_variants(self):
        gt = torch.rand(64, 64) * 60 + 2
        gt[:, :20] = 0                                             # 정답 없음
        gt[0, 30] = 95.0                                           # 상한 밖
        mask = C.eval_mask(gt)
        self.assertEqual(int(mask.sum()), 64 * 44 - 1)
        v = C.variants(gt * 0.5, gt, mask, C.FX["ms2"])
        self.assertAlmostEqual(v["raw"]["abs_rel"], 0.5, places=5)
        self.assertAlmostEqual(v["median"]["abs_rel"], 0.0, places=5)
        self.assertAlmostEqual(v["median"]["scale"], 2.0, places=5)
        self.assertAlmostEqual(v["focal"]["abs_rel"], 0.5, places=5)      # 같은 카메라면 focal = raw
        f = C.variants(gt, gt, mask, C.FX["fire"])["focal"]["abs_rel"]
        self.assertAlmostEqual(f, C.FX["fire"] / C.FX_TRAIN - 1, places=3)
        self.assertIsNone(C.variants(gt, gt, torch.zeros_like(mask), C.FX["ms2"]))
        self.assertLessEqual(C.variants(gt * 5, gt, mask, C.FX["ms2"])["raw"]["rmse"], C.MAX_DEPTH)   # 예측은 상한에서 잘림

    def test_silog(self):
        gt, pred = torch.rand(4, 32, 32) * 50 + 1, torch.rand(4, 32, 32) * 50 + 1
        mask = gt > 10
        d = torch.log(gt[mask]) - torch.log(pred[mask])            # 공식 util/loss.py 의 식
        self.assertAlmostEqual(C.silog(pred, gt, mask).item(), torch.sqrt(d.pow(2).mean() - 0.5 * d.mean().pow(2)).item(), places=6)

    def test_tone(self):
        rng = np.random.default_rng(1)
        raw = rng.integers(3000, 6000, (256, 640)).astype(np.uint16)
        x = D.tone(raw)
        self.assertEqual(x.dtype, np.float32)
        self.assertTrue(x.min() == 0.0 and x.max() == 1.0)
        self.assertAlmostEqual(float((x == 0).mean()), 0.01, delta=0.003)
        np.testing.assert_allclose(D.tone((raw.astype(np.float64) * 3 + 20000).astype(np.uint16)), x, atol=2e-3)   # 카메라의 값 범위 · 배율이 달라도 같은 입력

    def test_data(self):
        seqs = [s.strip() for s in open(os.path.join(HERE, "..", "splits", "ms2", "train_list.txt")) if s.strip()]
        keys = [k for k, _, _ in D.frames("train")]
        self.assertEqual(keys, [f"{s}/{n:06d}.png" for s in seqs[:2] for n in (3, 5, 7)])          # 목록 순서 → 이름순
        self.assertEqual([k for k, _, _ in D.frames("fire")], [f"{d}/{n:06d}.png" for d in ("frick_9_test", "hawkins_9_test") for n in (1, 2)])
        ev = D.Thermal("train")
        x0, d0, i0 = ev[1]
        self.assertEqual((tuple(x0.shape), tuple(d0.shape), i0), ((3, 256, 640), (256, 640), 1))
        g = x0 * torch.from_numpy(D.STD) + torch.from_numpy(D.MEAN)                                  # 정규화를 되돌리면 세 채널이 같은 [0, 1] 영상
        self.assertTrue(torch.allclose(g[0], g[1], atol=1e-5) and torch.allclose(g[0], g[2], atol=1e-5) and -1e-5 <= float(g.min()) and float(g.max()) <= 1 + 1e-5)
        self.assertAlmostEqual(float(d0.max()), float(np.array(Image.open(ev.items[1][2])).max()) / 256, places=4)
        tr = D.Thermal("train", aug="base", seed=3)
        flips = 0
        for ep in range(6):
            tr.epoch = ep
            x, d, _ = tr[1]
            a, b = torch.equal(x, x0) and torch.equal(d, d0), torch.equal(x, x0.flip(-1)) and torch.equal(d, d0.flip(-1))
            self.assertTrue(a or b)                                # 사진과 정답이 함께 뒤집힌다
            flips += b
            x2, d2, _ = tr[1]
            self.assertTrue(torch.equal(x, x2))                    # 같은 (시드, epoch, 번호) 면 같은 증강
        self.assertTrue(0 < flips < 6)
        self.assertEqual(len(D.Thermal("train", limit=4)), 4)
        self.assertEqual(tuple(D.Thermal("fire")[0][0].shape), (3, 512, 640))
        v = D.Thermal("vivid")
        self.assertEqual([k for k, _, _ in v.items], [f"outdoor_robust_night{i}/{n:06d}.png" for i in (1, 2) for n in (2, 4)])
        self.assertEqual((v.fx_key, v.border, D.Thermal("fire").border, len(D.Thermal("vivid_day"))), ("vivid", True, False, 2))
        vi, st = D.Thermal("vivid_in"), D.Thermal("sthereo")
        self.assertEqual([k for k, _, _ in vi.items], [f"{q}/{n:06d}.png" for q in ("indoor_robust_dark", "indoor_aggresive_local") for n in (2, 4)])   # 저자 test 목록의 순서
        self.assertEqual((vi.fx_key, vi.border, st.fx_key, st.border), ("vivid", True, "sthereo", False))
        self.assertEqual([k for k, _, _ in st.items], [f"sthereo_09_valley_evening/163007370000000000{n}.png" for n in (1, 2)])
        self.assertEqual((tuple(st[0][0].shape), tuple(st[0][1].shape), C.FX["sthereo"]), ((3, 512, 640), (512, 640), 429.433))

    def test_pairs_extra(self):
        f, t = D.Pairs(["freiburg"]), D.Pairs(["tartan"])
        self.assertEqual((len(f), len(t)), (1, 1))
        x, r = f[0]
        self.assertEqual((tuple(x.shape), tuple(r.shape), x.dtype, r.dtype), ((3, 298, 640), (3, 298, 640), torch.float32, torch.float32))
        self.assertTrue(torch.equal(x[0] * 0.229 + 0.485, x[1] * 0.224 + 0.456) or torch.allclose(x[0] * 0.229 + 0.485, x[1] * 0.224 + 0.456, atol=1e-5))   # 열화상은 세 채널이 같은 값
        x, r = t[0]
        self.assertEqual((tuple(x.shape), tuple(r.shape)), ((3, 512, 640), (3, 512, 640)))
        self.assertTrue(torch.isfinite(x).all() and torch.isfinite(r).all())

    def test_pair_labels(self):
        ds = D.PairLabels(["kaist", "llvip", "tartan"])
        self.assertEqual([(n, k) for n, k, _ in ds.items], [("kaist", "set00/V000/I00010.jpg"), ("kaist", "set03/V001/I00000.jpg"), ("llvip", "010001.jpg"),
                                                           ("tartan", "day1/20250822_115703/00000010.png")])
        for i in range(len(ds)):
            x, lab = ds[i]
            self.assertEqual((tuple(x.shape), tuple(lab.shape), x.dtype, lab.dtype), ((3, 512, 640), (512, 640), torch.float32, torch.float32))
            self.assertTrue(float(lab[:64].max()) == 0 and 3 <= float(lab[64:].min()) and float(lab.max()) <= 40)     # 위쪽 띠는 못 쓰는 화소 (0)
        f = D.PairLabels(["freiburg"])
        self.assertEqual((len(f), tuple(f[0][0].shape), tuple(f[0][1].shape)), (1, (3, 298, 640), (298, 640)))
        self.assertEqual(D.label_path("kaist", "set00/V000/I00010.jpg", "/r"), "/r/labels/kaist/set00/V000/I00010.png")
        bad = D.label_path("llvip", "010001.jpg")
        keep = open(bad, "rb").read()
        Image.fromarray(np.zeros((10, 10), np.uint16)).save(bad)                # 크기가 다른 라벨은 읽지 않는다
        with self.assertRaises(ValueError):
            D.PairLabels(["llvip"])[0]
        open(bad, "wb").write(keep)

    def test_labels(self):
        import labels as L

        class Fake:                                                 # UniDepthV2 자리에: 화면 왼쪽 → 오른쪽으로 0.05 – 300 m
            def infer(self, x8):
                d = torch.linspace(0.05, 300.0, x8.shape[-1]).expand(x8.shape[0], 1, x8.shape[-2], -1).clone()
                d[:, :, 0, 5] = float("nan")
                return {"depth": d}
        x8 = torch.full((2, 3, 40, 64), 200, dtype=torch.uint8)
        x8[:, :, 30:] = 0                                              # 아래쪽은 어두움 → 못 쓰는 화소
        lab = L.label_batch(Fake(), x8)
        self.assertEqual((tuple(lab.shape), lab.dtype), ((2, 40, 64), torch.float32))
        ok = lab > 0
        self.assertTrue(not ok[:, 34:].any() and not ok[:, 0, 5].any())             # 어두운 곳 · 유한하지 않은 값
        self.assertTrue(float(lab[ok].min()) >= C.PL_MIN and float(lab[ok].max()) <= C.PL_MAX and 0.5 < float(ok[:, :20].float().mean()) < 0.75)   # 0.1–200 m 만

    def test_shape_scale(self):
        g = torch.Generator().manual_seed(1)
        zn = torch.randn(3, 20, 30, generator=g) * 0.5
        zn = zn - zn.flatten(1).median(1).values[:, None, None]
        s0 = torch.tensor([2.0, 2.5, 3.0])
        logt = zn + s0[:, None, None]                                  # 목표 = 학생의 모양 그대로 · 스케일만 다름
        m = torch.rand(3, 20, 30, generator=g) < 0.5
        sh, sc, n = C.shape_scale(zn, s0, logt, m)
        self.assertEqual(n, 3)
        self.assertTrue(float(sh) < 1e-5 and float(sc) < 1e-5)
        sh, sc, _ = C.shape_scale(zn, s0 + torch.tensor([0.5, -0.5, 0.2]), logt, m)
        self.assertTrue(float(sh) < 1e-5)
        self.assertAlmostEqual(float(sc), 0.4, places=5)               # 스케일만 틀림 → 스케일 손실만 (|0.5| + |0.5| + |0.2|) / 3
        noise = torch.randn(3, 20, 30, generator=g) * 0.3
        a = C.shape_scale(zn + noise, s0, logt, m)
        b = C.shape_scale(zn + noise + 1.7, s0, logt, m)               # 모양에 상수를 더해도 모양 손실은 그대로 (스케일에 무관)
        self.assertTrue(float(a[0]) > 0.2)
        self.assertAlmostEqual(float(a[0]), float(b[0]), places=5)
        self.assertTrue(float(a[1]) < 0.05 and abs(float(b[1]) - 1.7) < 0.05)        # 그만큼 스케일의 목표가 옮겨 간다
        bad = logt.clone()
        bad[:, :3] += 5.0                                              # 선생이 크게 틀린 화소 15 %
        t0, t2 = C.shape_scale(zn, s0, bad, torch.ones_like(m), trim=0.0), C.shape_scale(zn, s0, bad, torch.ones_like(m), trim=0.2)
        self.assertTrue(float(t0[0]) > 1.0 and float(t2[0]) < 1e-5 and float(t2[1]) < 1e-5)     # 큰 쪽 20 % 를 버리면 남지 않는다 · 중앙값은 안 흔들린다
        few = torch.zeros_like(m)
        few[:, 0, :5] = True
        z = zn.clone().requires_grad_(True)
        sh, sc, n = C.shape_scale(z, s0, logt, few, min_px=100)
        self.assertEqual((n, float(sh), float(sc)), (0, 0.0, 0.0))
        (sh + sc).backward()                                           # 쓸 사진이 없어도 기울기가 끊기지 않는다
        z, sv = (zn + noise).requires_grad_(True), (s0 + 0.3).requires_grad_(True)
        sh, sc, _ = C.shape_scale(z, sv, logt, m)
        (sh + sc).backward()
        self.assertTrue(torch.allclose(sv.grad, torch.full((3,), 1 / 3)))                # 스케일 손실의 기울기 = 부호 / 사진 수
        self.assertTrue(float(z.grad.abs().sum()) > 0 and float(z.grad[~m].abs().sum()) == 0)

    def test_sep_model(self):
        M = self.M
        torch.manual_seed(0)
        sep = M.Sep(M.build("vits")).eval()
        x = torch.randn(2, 3, 256, 640)
        with torch.no_grad():
            y, sc = sep(torch.nn.functional.interpolate(x, M.net_size(256, 640)))
            zn, s2 = M.predict_sep(sep, x, (256, 640))
            p = M.predict(sep, x, (256, 640))
        self.assertEqual((tuple(y.shape), tuple(sc.shape), tuple(zn.shape), tuple(p.shape)), ((2, 252, 644), (2,), (2, 256, 640), (2, 256, 640)))
        self.assertTrue(torch.allclose(sc, torch.full((2,), C.S0)) and torch.allclose(s2, sc))           # 처음에는 모든 사진에 S0
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):                                # 혼합 정밀도에서도 모양 · 스케일은 fp32 로 나온다
            yb, sb = sep(torch.nn.functional.interpolate(x[:1], M.net_size(256, 640)))
        self.assertEqual((yb.dtype, sb.dtype), (torch.float32, torch.float32))
        self.assertTrue(float(zn.flatten(1).median(1).values.abs().max()) < 0.05)                        # 모양의 중앙값은 0 (올린 뒤라 조금 어긋난다)
        self.assertTrue(torch.allclose(p, M.sep_depth(zn, s2)) and C.MIN_DEPTH <= float(p.min()) and float(p.max()) <= C.MAX_DEPTH)
        self.assertTrue(abs(float(p.median()) - float(np.exp(C.S0))) < 2.0)                              # 처음의 깊이는 15 m 안팎
        names = [n for n, _ in sep.named_parameters()]
        self.assertTrue(any("pretrained" in n for n in names) and any(n.startswith("scale_head.") for n in names) and any(n.startswith("net.depth_head.") for n in names))
        c = M.count_params(sep)
        self.assertTrue(20 < c["encoder_m"] < 25 and 0.09 < c["scale_head_m"] < 0.11)                    # 384 → 256 → 1
        with tempfile.TemporaryDirectory() as t:                           # 평가는 저장된 모델의 구조를 스스로 알아본다
            torch.save(sep.state_dict(), os.path.join(t, "sep.pt"))
            torch.save(self.model.state_dict(), os.path.join(t, "base.pt"))
            a, b = M.load("vits", os.path.join(t, "sep.pt")).eval(), M.load("vits", os.path.join(t, "base.pt"))
            self.assertTrue(isinstance(a, M.Sep) and not isinstance(b, M.Sep))
            with torch.no_grad():
                self.assertTrue(torch.allclose(M.predict(a, x, (256, 640)), p, atol=1e-5))
        # 한 걸음 학습: 모양 · 스케일 손실의 기울기가 인코더 · 헤드 · 스케일 헤드 모두에 닿는다
        sep.train()
        gt = torch.rand(2, 256, 640) * 40 + 2
        zn, sc = M.predict_sep(sep, x, (256, 640))
        sh, sl, _ = C.shape_scale(zn, sc, torch.log(gt), torch.rand(2, 256, 640) < 0.1)
        (sh + sl).backward()
        gr = {n: p_.grad is not None and float(p_.grad.abs().sum()) > 0 for n, p_ in sep.named_parameters()}
        self.assertTrue(gr["scale_head.2.bias"] and any(v for n, v in gr.items() if n.startswith("net.depth_head.")) and any(v for n, v in gr.items() if "pretrained" in n))

    def test_pairs(self):
        ds = D.Pairs(["kaist", "llvip"])
        self.assertEqual([k for k, _, _ in ds.items], ["set00/V000/I00010.jpg", "set03/V001/I00000.jpg", "010001.jpg"])
        for i in range(3):
            x, rgb = ds[i]
            self.assertEqual((tuple(x.shape), tuple(rgb.shape), x.dtype, rgb.dtype), ((3, 512, 640), (3, 512, 640), torch.float32, torch.float32))
        v = D.Pairs(["vivid_day"])
        self.assertEqual(([k for k, _, _ in v.items], tuple(v[0][1].shape)), (["outdoor_robust_day1/000002.png", "outdoor_robust_day1/000004.png"], (3, 480, 640)))
        self.assertEqual([k for k, _, _ in v.items], [k for k, _, _ in D.Thermal("vivid_day").items])          # 선생 점검이 같은 순서에 기댄다

    def test_ssi_trim(self):
        torch.manual_seed(1)
        disp = torch.rand(2, 64, 80) + 0.05
        mask = torch.ones(2, 64, 80, dtype=torch.bool)
        pred = 1.0 / (3.0 * disp + 0.2)                            # 학생의 역깊이가 선생 값의 일차식이면 손실 0 (축척 · 이동 무관)
        self.assertLess(C.ssi_trim(pred, disp, mask).item(), 1e-5)
        self.assertLess(C.ssi_trim(pred, disp * 7 + 2, mask).item(), 1e-5)
        bad = pred.clone()
        bad[:, :6] = 50.0                                          # 사진의 9 % 가 엉뚱한 값 → 큰 오차 20 % 를 버리므로 손실은 작게 남고, 안 버리면 커진다
        self.assertLess(C.ssi_trim(bad, disp, mask).item(), C.ssi_trim(bad, disp, mask, trim=0.0).item() * 0.5)
        self.assertGreater(C.ssi_trim(1.0 / (torch.rand(2, 64, 80) + 0.05), disp, mask).item(), 0.3)   # 무관한 예측은 손실이 큼
        self.assertEqual(C.ssi_trim(pred, disp, torch.zeros_like(mask)).item(), 0.0)                     # 쓸 픽셀이 없으면 0
        p = pred.clone().requires_grad_(True)
        C.ssi_trim(p * 1.3, disp.flip(-1), mask).backward()
        self.assertTrue(torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0)

    def test_vivid_grid(self):
        grid, inside = C.vivid_grid()
        self.assertEqual((tuple(grid.shape), tuple(inside.shape)), ((1, 512, 640, 2), (512, 640)))
        self.assertTrue(0.55 < inside.float().mean().item() < 0.80)                 # 열화상이 더 넓게 본다 → 가장자리는 RGB 밖
        self.assertTrue(inside[256, 320] and not inside[5, 320] and not inside[256, 5])
        u = (grid[0, 256, 320, 0].item() + 1) / 2 * 639                              # 열화상 가운데는 RGB 가운데 근처로 간다
        v = (grid[0, 256, 320, 1].item() + 1) / 2 * 479
        self.assertTrue(abs(u - 322) < 12 and abs(v - 240) < 12)
        z0 = 15.0                                                                    # 직접 계산과 대조 (한 픽셀)
        x = np.array([(100 - C.VIVID_K_T[0][2]) / C.VIVID_K_T[0][0] * z0, (300 - C.VIVID_K_T[1][2]) / C.VIVID_K_T[1][1] * z0, z0]) @ np.array(C.VIVID_R).T + np.array(C.VIVID_T)
        ur, vr = C.VIVID_K_RGB[0][0] * x[0] / x[2] + C.VIVID_K_RGB[0][2], C.VIVID_K_RGB[1][1] * x[1] / x[2] + C.VIVID_K_RGB[1][2]
        self.assertAlmostEqual((grid[0, 300, 100, 0].item() + 1) / 2 * 639, ur, places=2)
        self.assertAlmostEqual((grid[0, 300, 100, 1].item() + 1) / 2 * 479, vr, places=2)

    def test_teacher(self):
        import train as T
        t = self.M.build_teacher("vits")
        self.assertTrue(isinstance(t.depth_head.scratch.output_conv2[-1], torch.nn.ReLU) and not any(p.requires_grad for p in t.parameters()))
        rgb = torch.randn(1, 3, 480, 640)
        d = self.M.teacher_disp(t, rgb)
        self.assertTrue(tuple(d.shape) == (1, 480, 640) and float(d.min()) >= 0)
        grid, inside = C.vivid_grid()
        dw, ok = T.pseudo(t, rgb, False, grid, inside)
        self.assertEqual((tuple(dw.shape), tuple(ok.shape), ok.dtype), ((1, 512, 640), (1, 512, 640), torch.bool))
        self.assertFalse(ok[0][~inside].any())
        dark = (torch.zeros(1, 3, 512, 640) - torch.from_numpy(D.MEAN)) / torch.from_numpy(D.STD)   # 검은 RGB → 쓸 픽셀 없음
        self.assertEqual(int(T.bright(dark).sum()), 0)
        self.assertEqual(int(T.bright(dark + 1.0).sum()), 512 * 640)
        c = T.teacher_check(t, torch.device("cpu"), False)                           # 가짜 자료라 값은 뜻 없음 — 흐름과 장 수만
        self.assertTrue(c is None or c["n"] <= 2)

    def test_border(self):
        gt = torch.zeros(512, 640)
        gt[::9, ::7] = 20.0                                        # LiDAR 처럼 드문 정답: 모든 줄 · 열에서 값 없는 비율 > 0.4 → 변마다 101 까지 들어간다
        self.assertEqual(C.border_box(gt), (101, 512 - 101, 101, 640 - 101))
        m = C.eval_mask(gt, border=True)
        self.assertEqual(int(m.sum()), int((gt[101:411, 101:539] > 0).sum()))
        self.assertGreater(int(C.eval_mask(gt).sum()), int(m.sum()))
        dense = torch.full((512, 640), 10.0)                       # 빽빽한 정답: 최소 테두리 40 만 뺀다
        self.assertEqual(C.border_box(dense), (40, 472, 40, 600))
        half = dense.clone()
        half[:70] = 0                                              # 위 70 줄이 비어 있으면 거기까지 들어간다
        self.assertEqual(C.border_box(half)[0], 70)

    def test_model(self):
        M = self.M
        self.assertEqual((M.net_size(256, 640), M.net_size(512, 640)), ((252, 644), (518, 644)))
        with torch.no_grad():
            p = M.predict(self.model, torch.randn(2, 3, 256, 640), (256, 640))
        self.assertEqual(tuple(p.shape), (2, 256, 640))
        self.assertTrue(p.dtype == torch.float32 and 0 <= float(p.min()) and float(p.max()) <= C.MAX_DEPTH)
        c = M.count_params(self.model)
        self.assertTrue(20 < c["encoder_m"] < 25 and c["head_m"] > 1)
        names = [n for n, _ in self.model.named_parameters()]
        self.assertTrue(any("pretrained" in n for n in names) and any(n.startswith("depth_head.") for n in names))

    def test_eval(self):
        from evaluate import eval_set
        with tempfile.TemporaryDirectory() as out:
            s, rows = eval_set(self.model, "fire", torch.device("cpu"), bs=2, workers=0, samples=2, out=out)
            self.assertEqual((s["n"], s["n_all"], len(rows)), (4, 4, 4))
            self.assertEqual((s["day"]["n"], s["night"]["n"]), (2, 2))
            self.assertAlmostEqual(s["raw"]["abs_rel"], float(np.mean([r["raw"]["abs_rel"] for r in rows])), places=9)
            pngs = sorted(os.listdir(os.path.join(out, "samples")))
            self.assertEqual(len(pngs), 2)
            self.assertEqual(np.array(Image.open(os.path.join(out, "samples", pngs[0]))).shape, (512, 640))
        s, _ = eval_set(self.model, "test_night", torch.device("cpu"), bs=3, workers=0)
        self.assertEqual(s["n"], 3)
        s, _ = eval_set(self.model, "vivid", torch.device("cpu"), bs=2, workers=0)
        self.assertEqual((s["n"], s["n_all"]), (4, 4))
        for name, n in (("vivid_in", 4), ("sthereo", 2)):
            s, _ = eval_set(self.model, name, torch.device("cpu"), bs=2, workers=0)
            self.assertEqual((s["n"], s["n_all"]), (n, n))
        from evaluate import TEST_SETS
        self.assertEqual(set(TEST_SETS), set(D.SETS) - {"train", "val"})

    def test_report(self):
        import contextlib
        import shutil
        import report as R

        def met(a):
            return {v: {k: a for k in C.METRICS} for v in C.VARIANTS} | {"n": 1, "n_all": 1, "scale": {"abs_med": a / 2, "bias": -a / 2}}

        def run(t, arm, s, ms2, zs):
            d = os.path.join(t, "runs", f"{arm}_s{s}", "eval")
            os.makedirs(d, exist_ok=True)
            m = {k: met(ms2) for k in ("test_day", "test_night", "test_rainy")} | {k: met(v) for k, v in zs.items()} | {"vivid_day": met(0.150)}
            m["fire"] |= {"day": met(zs["fire"]), "night": met(zs["fire"])}
            json.dump(m, open(os.path.join(d, "metrics.json"), "w"))

        def summ(t):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                R.summary(os.path.join(t, "runs"), os.path.join(t, "summary"))
            return buf.getvalue().splitlines(), json.load(open(os.path.join(t, "summary", "summary.json")))["judge"]

        base = lambda s: {"vivid": 0.160 + 0.004 * s, "fire": 0.300 + 0.005 * s, "vivid_in": 0.400 + 0.010 * s, "sthereo": 0.200 + 0.002 * s}  # noqa: E731
        with tempfile.TemporaryDirectory() as t:
            for s in range(3):
                b = base(s)
                run(t, "base", s, 0.075 + 0.001 * s, b)
                run(t, "sep", s, 0.0755 + 0.001 * s, {k: v + 0.001 for k, v in b.items()})
                run(t, "ours", s, 0.076 + 0.001 * s, {"vivid": b["vivid"] - 0.010, "fire": b["fire"] - 0.100, "vivid_in": b["vivid_in"] - 0.100, "sthereo": b["sthereo"] + 0.001})
            out, j = summ(t)
            self.assertLessEqual(len(out), 46)                     # run.sh 가 남기는 끝 50 줄 안에 들어가야 한다 (zip · 끝 줄 포함)
            self.assertTrue(all(len(x) < 400 for x in out))
            self.assertEqual([x.split()[1] for x in out if x.startswith("[표2] ") and "|" in x and not x.startswith("[표2] 처음")], ["base", "sep", "ours"])
            self.assertTrue(j["gate"]["ok"] and all(j["gap"][k]["ok"] for k in ("vivid", "fire", "vivid_in", "sthereo")))
            tg = j["ours"]["targets"]
            self.assertEqual([tg[k]["verdict"] for k in ("vivid", "fire", "vivid_in", "sthereo")], ["개선", "개선", "개선", "차이 없음"])
            self.assertAlmostEqual(tg["fire"]["noise"], 0.0256)    # 같은 작업의 범위 0.010 보다 앞선 실험의 범위가 넓다 → 넓은 쪽
            self.assertAlmostEqual(tg["vivid"]["noise"], 0.008)    # 같은 작업의 범위가 더 넓다
            self.assertAlmostEqual(tg["vivid_in"]["noise"], 0.020)  # 새 대상: 같은 작업의 범위 (MS2 의 바닥 0.0013 보다 넓다)
            self.assertTrue(j["ours"]["ms2_kept"] and j["ours"]["claim"])                    # MS2 +0.001 ≤ 0.002 · 개선 3 곳 · 나빠진 곳 없음
            self.assertEqual(([v["verdict"] for v in j["sep"]["targets"].values()], j["sep"]["ms2_kept"], "claim" in j["sep"]), (["차이 없음"] * 4, True, False))   # sep 은 MS2 유지만 본다
            self.assertTrue(any("구조를 바꿔도 MS2 점수 유지" in x for x in out if x.startswith("[판정] sep")))
            self.assertTrue(any(x.startswith("[표4] base  | 0.0380") for x in out))          # MS2 평균 (0.075 + 0.076 + 0.077) / 3 / 2
            self.assertTrue(any("여러 곳에서 개선" in x for x in out if x.startswith("[판정] ours")))
            for s in range(3):                                     # 한 곳에서 나빠지면 주장이 서지 않는다
                run(t, "ours", s, 0.076 + 0.001 * s, base(s) | {"vivid": base(s)["vivid"] - 0.010, "vivid_in": base(s)["vivid_in"] - 0.1, "fire": base(s)["fire"] + 0.040})
            _, j = summ(t)
            self.assertEqual((j["ours"]["targets"]["fire"]["verdict"], j["ours"]["claim"]), ("나빠짐", False))
            for s in range(3):                                     # MS2 가 잡음 폭보다 더 나빠져도 서지 않는다
                run(t, "ours", s, 0.080 + 0.001 * s, {k: v - 0.1 for k, v in base(s).items()})
            _, j = summ(t)
            self.assertEqual((j["ours"]["ms2_kept"], j["ours"]["claim"]), (False, False))
            for s in (1, 2):
                shutil.rmtree(os.path.join(t, "runs", f"ours_s{s}"))
            out, j = summ(t)
            self.assertTrue(j["ours"]["targets"]["fire"]["verdict"] == "판정 안 함" and "claim" not in j["ours"])
            self.assertTrue(any("시드가 하나" in x for x in out))
            # 이 작업에 base 가 없으면 앞선 작업의 base (setup/base_ref.json 꼴: 시드별 raw AbsRel 만, 새 대상은 없음) 를 기준으로 판정한다
            for s in range(3):
                shutil.rmtree(os.path.join(t, "runs", f"base_s{s}"))
                shutil.rmtree(os.path.join(t, "runs", f"sep_s{s}"))
                run(t, "pl", s, 0.075, {"fire": 0.20 + 0.002 * s, "vivid": 0.12, "vivid_in": 0.3, "sthereo": 0.2})
            shutil.rmtree(os.path.join(t, "runs", "ours_s0"))
            ref = {"source": "앞선 작업", "seeds": {str(s): {k: {"raw": {"abs_rel": v}} for k, v in
                   (("test_day", 0.075), ("test_night", 0.075 + 0.001 * s), ("test_rainy", 0.075), ("fire", 0.30 + 0.005 * s), ("vivid", 0.16 + 0.004 * s))} for s in range(3)}}
            old, R.BASE_REF = R.BASE_REF, os.path.join(t, "base_ref.json")
            json.dump(ref, open(R.BASE_REF, "w"))
            out, j = summ(t)
            R.BASE_REF = old
            self.assertEqual(({k: v["verdict"] for k, v in j["pl"]["targets"].items()}, "claim" in j["pl"]), ({"vivid": "개선", "fire": "개선"}, False))   # 대상 둘만 견줄 수 있으면 '여러 곳' 은 말하지 않는다
            self.assertTrue(any("2 곳만 견줄 수 있어" in x for x in out))
            self.assertTrue(any("(앞선 작업)" in x for x in out))
            for s in range(3):
                shutil.rmtree(os.path.join(t, "runs", f"pl_s{s}"))
            with tempfile.TemporaryDirectory() as t2:              # 단계 2: base · tr 을 MS2 만으로 — val 의 마지막 세 epoch 으로 구조 점검
                for arm, vals in (("base", (0.090, 0.080, 0.078, 0.082)), ("tr", (0.095, 0.086, 0.084, 0.085))):
                    d = os.path.join(t2, "runs", f"{arm}_s0", "eval")
                    os.makedirs(d)
                    json.dump({k: met(0.075) for k in ("test_day", "test_night", "test_rainy")}, open(os.path.join(d, "metrics.json"), "w"))
                    with open(os.path.join(t2, "runs", f"{arm}_s0", "val_log.csv"), "w") as f:
                        f.write("epoch,loss,loss_pl,val_abs_rel,val_rmse,val_d1,val_n\n" + "".join(f"{i + 1},0.1,,{v},2.5,0.95,16\n" for i, v in enumerate(vals)))
                out2, j2 = summ(t2)
                self.assertAlmostEqual(j2["tr_structure"]["tr"], (0.086 + 0.084 + 0.085) / 3)
                self.assertTrue(j2["tr_structure"]["ok"])                   # 0.0850 ≤ 0.0800 + 0.005
                self.assertTrue(any(x.startswith("[판정] 구조 점검") and "통과" in x for x in out2))
                self.assertFalse(any("raw AbsRel:" in x for x in out2))    # 처음 보는 대상을 재지 않았으면 그 줄은 없다
                os.makedirs(os.path.join(t2, "runs", "rt_s0", "eval"))         # rt: 1 단계의 val 도 표에 · 구조 점검도 같은 기준
                os.makedirs(os.path.join(t2, "runs", "rt_s0", "rel"))
                json.dump({k: met(0.075) for k in ("test_day", "test_night", "test_rainy")}, open(os.path.join(t2, "runs", "rt_s0", "eval", "metrics.json"), "w"))
                json.dump({"val_aligned_abs_rel": 0.071}, open(os.path.join(t2, "runs", "rt_s0", "rel", "done.json"), "w"))
                with open(os.path.join(t2, "runs", "rt_s0", "val_log.csv"), "w") as f:
                    f.write("epoch,loss,loss_pl,val_abs_rel,val_rmse,val_d1,val_n\n1,0.1,,0.10,2.5,0.9,16\n2,0.1,,0.09,2.5,0.9,16\n")
                out3, j3 = summ(t2)
                self.assertAlmostEqual(j3["rt_structure"]["rt"], 0.095)
                self.assertFalse(j3["rt_structure"]["ok"])                   # 0.095 > 0.0800 + 0.005
                self.assertTrue(any("rt 의 상대 깊이 모델 (사진마다 맞춘 뒤) 0.0710" in x for x in out3))
            for arm in ("base", "sep", "ours", "plv"):             # 팔 넷 × 시드 셋도 끝 50 줄 안에 든다 (zip · 끝 줄 포함)
                for s in range(3):
                    run(t, arm, s, 0.075, base(0))
            out, _ = summ(t)
            self.assertLessEqual(len(out), 46)
            self.assertTrue(all(len(x) < 400 for x in out))

    def test_tr_model(self):
        M = self.M
        torch.manual_seed(0)
        tr = M.build_tr("vits")
        names = {n for n, p in tr.named_parameters() if p.requires_grad}
        self.assertTrue(names and all(n.startswith("scalemap.") for n in names))          # 학습하는 것은 모듈뿐
        self.assertEqual(M.count_params(tr)["trainable_m"], round(sum(p.numel() for p in tr.scalemap.parameters()) / 1e6, 2))
        tr.train()
        self.assertFalse(tr.rel.training)                                                  # 고정한 쪽은 늘 eval
        x = torch.randn(2, 3, 256, 640)
        d, a, b, dr, xn = M.predict_tr(tr, x, (256, 640))
        self.assertTrue(torch.equal(a, torch.full_like(a, 0.01)) and torch.equal(b, torch.full_like(b, 0.01)))   # 처음 맵은 모든 화소에서 0.01
        self.assertEqual((tuple(d.shape), tuple(a.shape), tuple(b.shape), tuple(dr.shape), tuple(xn.shape)),
                         ((2, 256, 640), (2, 252, 644), (2, 252, 644), (2, 252, 644), (2, 3, 252, 644)))
        self.assertTrue(float(a.min()) > 0 and float(b.min()) > 0 and float(dr.min()) >= 0)
        self.assertTrue(float(d.min()) > 0 and float(d.max()) <= 100.0 + 1e-3)                  # 1 / (0.01 · D_r + 0.01) ≤ 100 m, 자르지 않는다
        self.assertTrue(torch.allclose(d, M.predict(tr, x, (256, 640))))
        gt = torch.rand(2, 256, 640) * 40 + 2
        C.silog(d, gt, gt > 0).backward()                                                  # 기울기는 모듈에만
        self.assertTrue(all(p.grad is None for p in tr.rel.parameters()) and any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in tr.scalemap.parameters()))
        self.assertTrue(float(tr.scalemap.s2.m[-1].bias.grad.abs()) > 0 and float(tr.scalemap.t2.m[-1].bias.grad.abs()) > 0)   # 0 으로 시작한 마지막 층에도 기울기
        with tempfile.TemporaryDirectory() as t:                                           # 체크포인트 = 고정한 쪽까지 전부 → 평가가 스스로 TR 로 싣는다
            torch.save(tr.state_dict(), os.path.join(t, "tr.pt"))
            tr2 = M.load("vits", os.path.join(t, "tr.pt")).eval()
            self.assertIsInstance(tr2, M.TR)
            with torch.no_grad():
                self.assertTrue(torch.allclose(M.predict(tr2, x, (256, 640)), M.predict(tr.eval(), x, (256, 640)), atol=1e-5))

    def test_tr_losses(self):
        g = torch.Generator().manual_seed(2)
        dr = torch.rand(3, 20, 30, generator=g) * 4 + 0.5
        s0, t0 = torch.tensor([0.02, 0.05, 0.01]), torch.tensor([0.01, 0.002, 0.03])
        gt = 1.0 / (s0[:, None, None] * dr + t0[:, None, None])
        m = torch.rand(3, 20, 30, generator=g) < 0.4
        s, t = C.fit_inv(dr, gt, m)
        self.assertTrue(torch.allclose(s.float(), s0, atol=1e-5) and torch.allclose(t.float(), t0, atol=1e-5))   # 정확히 맞는 경우를 되찾는다
        a, b = torch.full((3, 10, 15), 0.03, requires_grad=True), torch.full((3, 10, 15), 0.01, requires_grad=True)
        loss, k = C.tr_pseudo(a, b, dr, gt, m)
        self.assertEqual(k, 3)
        exp = float(((0.03 - s0).abs().mean() + (0.01 - t0).abs().mean()))
        self.assertAlmostEqual(float(loss), exp, places=5)
        loss.backward()
        self.assertTrue(float(a.grad.abs().sum()) > 0)
        noisy = gt * torch.exp(torch.randn(3, 20, 30, generator=g) * 0.5)                   # 상대 깊이와 정답이 잘 안 맞으면 (δ1 < 0.99) 쓰지 않는다
        loss, k = C.tr_pseudo(a, b, dr, noisy, m)
        self.assertEqual((k, float(loss)), (0, 0.0))
        img = torch.randn(2, 3, 12, 16)
        self.assertAlmostEqual(float(C.edge_smooth(torch.full((2, 12, 16), 0.3), img)), 0.0, places=6)   # 고른 맵은 0
        ramp = torch.arange(16.0).expand(2, 12, 16) * 0.01
        self.assertAlmostEqual(float(C.edge_smooth(ramp, torch.zeros(2, 3, 12, 16))), 0.01, places=6)   # TR2M 식: 평균으로 나누지 않는다 (가로 차의 평균 0.01 + 세로 0)
        rough = torch.rand(2, 12, 16, generator=g) + 0.1
        self.assertTrue(float(C.edge_smooth(rough, torch.zeros(2, 3, 12, 16))) > float(C.edge_smooth(rough, img * 10)))   # 영상의 경계에서는 덜 벌한다

    def test_rel_train(self):
        M = self.M
        g = torch.Generator().manual_seed(3)
        gt = torch.rand(2, 20, 30, generator=g) * 40 + 2
        m = torch.rand(2, 20, 30, generator=g) < 0.5
        inv = 1.0 / gt
        self.assertLess(float(C.ssi_disp(inv * 7.0 + 0.3, gt, m)), 1e-5)          # 배율 · 기준점이 달라도 0
        p = (inv + torch.rand(2, 20, 30, generator=g) * 0.02).requires_grad_(True)
        loss = C.ssi_disp(p, gt, m)
        loss.backward()
        self.assertTrue(float(loss) > 0.01 and float(p.grad[m].abs().sum()) > 0 and float(p.grad[~m].abs().sum()) == 0)
        self.assertEqual(float(C.ssi_disp(p, gt, torch.zeros_like(m))), 0.0)     # 쓸 화소가 없으면 0
        torch.manual_seed(0)
        rel, teacher = M.build_rel("vits"), M.build_teacher("vits")
        self.assertTrue(all(q.requires_grad for q in rel.parameters()) and not any(q.requires_grad for q in teacher.parameters()))
        d = M.rel_forward(rel, torch.randn(2, 3, 256, 640), (256, 640))
        self.assertEqual(tuple(d.shape), (2, 256, 640))
        d.mean().backward()
        self.assertTrue(any(q.grad is not None and float(q.grad.abs().sum()) > 0 for n, q in rel.named_parameters() if "pretrained" in n))
        import train_rel                                                       # 1 단계를 끝까지 (가짜 묶음, 작은 인코더) → rel.pt 를 TR 이 싣는다
        with tempfile.TemporaryDirectory() as t:
            argv = sys.argv
            sys.argv = ["train_rel", "--encoder", "vits", "--epochs", "2", "--bs", "2", "--workers", "0", "--amp", "0", "--device", "cpu",
                        "--out", os.path.join(t, "rt_s0", "rel"), "--ckpt", os.path.join(t, "ck")]
            try:
                self.assertEqual(train_rel.main(), 0)
            finally:
                sys.argv = argv
            done = json.load(open(os.path.join(t, "rt_s0", "rel", "done.json")))
            self.assertEqual((done["steps"], done["stage"]), (6, "rel"))                  # 6 장 / 배치 2 × 2 epoch
            self.assertTrue(os.path.exists(os.path.join(t, "ck", "rel.pt")) and not os.path.exists(os.path.join(t, "ck", "last.pt")))
            with open(os.path.join(t, "rt_s0", "rel", "val_log.csv")) as f:
                self.assertEqual(len(f.read().strip().splitlines()), 3)
            tr = M.build_tr("vits", os.path.join(t, "ck", "rel.pt"))
            self.assertTrue(all(torch.equal(tr.rel.state_dict()[k], v) for k, v in torch.load(os.path.join(t, "ck", "rel.pt"), weights_only=True).items()))

    def test_rel_check(self):
        import rel_check as RC
        pick, ok = RC.verdict({"vitl": {"rgb": 0.10, "rgb_low": 0.15, "thr": 0.16}, "vits": {"rgb": 0.10, "rgb_low": 0.18, "thr": 0.21}})   # 견주는 것은 같은 높이의 RGB
        self.assertEqual((pick, ok), ("vitl", {"vitl": True, "vits": False}))
        pick, ok = RC.verdict({"vitl": {"rgb_low": 0.15, "thr": 0.1701}, "vits": {"rgb_low": 0.18, "thr": 0.19}})
        self.assertEqual(pick, "vits")                                   # 통과한 것 가운데 열화상 오차가 작은 쪽 (L 은 0.0201 차라 못 넘음)
        self.assertEqual(RC.verdict({"vitl": {"rgb_low": 0.10, "thr": 0.30}})[0], None)
        th = D.Thermal("val")                                            # 열화상 쪽은 학생과 같은 입력 · 같은 순서
        self.assertEqual(len(th), 3)
        torch.manual_seed(0)
        teacher = self.M.build_teacher("vits").eval()
        rows = RC.measure(teacher, th, torch.device("cpu"), 2, 0)
        self.assertEqual([r[0] for r in rows], [0, 1, 2])
        for sc in (0.5, 2.0):                                            # 크기를 바꿔 넣어도 정답 크기에서 잰다
            r2 = RC.measure(teacher, th, torch.device("cpu"), 1, 0, sc)
            self.assertEqual([r[0] for r in r2], [0, 1, 2])
        self.assertTrue(all(0 <= r[1] < 10 and 0 <= r[2] <= 1 for r in rows))

    def test_unpack(self):
        import get_data as G
        with tempfile.TemporaryDirectory() as t:
            tar = os.path.join(t, "x.tar")
            good = ["sync_data/_2021-08-06-10-59-33/thr/img_left/000001.png", "proj_depth/_2021-08-06-10-59-33/thr/depth_filtered/000001.png",
                    "firestereo/frick_2_test/img_left/02250.png", "vivid/outdoor_robust_night1/Thermal/000001.png", "vivid/outdoor_robust_day2/Depth_T/000001.png",
                    "vivid/outdoor_robust_day1/RGB/000001.png", "pairs/kaist/set00/V000/lwir/I00010.jpg", "pairs/llvip/visible/010001.jpg",
                    "vivid/indoor_robust_dark/Thermal/000001.png", "vivid/indoor_aggresive_local/Depth_T/000321.png",
                    "sthereo/sthereo_09_valley_evening/Thermal/1630073626519169347.png", "sthereo/sthereo_09_valley_evening/Depth_T/1630073626519169347.png",
                    "sync_data/_2021-08-06-10-59-33/rgb/img_left/000001.png", "proj_depth/_2021-08-06-10-59-33/rgb/depth_filtered/000001.png",   # RGB = 선생 점검용 묶음
                    "pairs/freiburg/seq_02_night/06/ir/1579632394_1905673020.png", "pairs/freiburg/seq_02_night/06/rgb/1579632394_1905673020.jpg",
                    "pairs/tartan/day4/20250828_000502/thermal/00000120.png", "pairs/tartan/day4/20250828_000502/rgb/00000120.jpg"]
            bad = ["../evil.png", "sync_data/_2021-08-06-10-59-33/calib.npy", "sync_data/../../x/thr/img_left/1.png", "firestereo/frick_2_train/img_left/1.png",
                   "sync_data/_2021-08-06-10-59-33/nir/img_left/000001.png", "/abs/sync_data/_1/thr/img_left/1.png", "vivid/indoor_robust_dark/RGB/000001.png", "sthereo/sthereo_09_valley_evening/Thermal/1.png", "sthereo/x/Thermal/1630073626519169347.png",
                   "vivid/outdoor_robust_night1/RGB/000001.png", "pairs/kaist/set00/V000/lwir/../../x.jpg", "pairs/other/visible/010001.jpg",
                   "pairs/freiburg/NOTICE.txt", "pairs/tartan/day3/20250826_125213/thermal/../x.png", "pairs/freiburg/seq_00_day/03/ir/a.jpg"]
            with tarfile.open(tar, "w") as tf:
                for n in good + bad:
                    ti = tarfile.TarInfo(n)
                    ti.size = 3
                    tf.addfile(ti, io.BytesIO(b"abc"))
                ti = tarfile.TarInfo("firestereo/frick_2_test/img_left/99999.png")       # 링크는 풀지 않는다
                ti.type, ti.linkname = tarfile.SYMTYPE, "/etc/passwd"
                tf.addfile(ti)
            dest = os.path.join(t, "out")
            self.assertEqual(G.unpack(tar, dest), len(good))
            found = sorted(os.path.relpath(os.path.join(d, f), dest) for d, _, fs in os.walk(dest) for f in fs)
            self.assertEqual(found, sorted(good))
            self.assertEqual(os.listdir(t).count("evil.png"), 0)


class TeacherCheck(unittest.TestCase):
    """선생 점검 (src/teacher_check.py) 의 계산 부품 — 모델 없이."""

    def test_align_recovers_shape(self):
        import teacher_check as T
        g = torch.Generator().manual_seed(0)
        gt = 2.0 + 30.0 * torch.rand(40, 60, generator=g)
        inv = 3.0 * (1.0 / gt) + 0.7                                  # 역깊이에 배율 · 기준점이 붙은 예측 → 맞추면 정답이 돼야 한다
        m = torch.ones_like(gt, dtype=torch.bool)
        self.assertLess((T.aligned(inv, gt, m) - gt.double()).abs().max().item(), 1e-3)             # 입력이 float32 라 1e-5 쯤의 오차는 남는다
        a, b = T.fit_ss(torch.arange(10.0).double(), 2.0 * torch.arange(10.0).double() - 1.0)
        self.assertAlmostEqual(a.item(), 2.0, places=9)
        self.assertAlmostEqual(b.item(), -1.0, places=9)

    def test_rank_measures(self):
        import teacher_check as T
        x = np.arange(100.0)
        self.assertAlmostEqual(T.spearman(x, x ** 3), 1.0, places=9)
        self.assertAlmostEqual(T.spearman(x, -x), -1.0, places=9)
        self.assertTrue(np.isnan(T.spearman(x, np.ones(100))))
        self.assertAlmostEqual(T.keep_half(x, x), np.mean(x[:50]) / np.mean(x), places=9)      # 점수가 오차와 같으면 믿는 절반 = 작은 절반
        self.assertAlmostEqual(T.keep_half(-x, x), np.mean(x[50:]) / np.mean(x), places=9)     # 거꾸로면 큰 절반
        self.assertEqual(T.sign_fit(x, -x), -1.0)
        rho, keep = T.per_image(np.tile(x, 2), np.tile(x, 2), np.repeat([0, 1], 100))
        self.assertAlmostEqual(rho, 1.0, places=9)
        self.assertLess(keep, 0.6)

    def test_ridge_learns_direction(self):
        import teacher_check as T
        rng = np.random.default_rng(0)
        X = rng.normal(size=(4000, 3))
        y = 1.5 * X[:, 0] - 0.5 * X[:, 1] ** 2 + 0.05 * rng.normal(size=4000)                 # 셋째 단서는 무관
        f = T.ridge_fit(X[:3000], y[:3000])
        self.assertGreater(T.spearman(f(X[3000:]), y[3000:]), 0.95)

    def test_tables_and_verdict(self):
        import teacher_check as T
        rng = np.random.default_rng(1)
        rows, pxs, img = [], [], []
        for i in range(80):
            lum = 0.1 if i % 2 else 0.5                                                       # 밝은 사진 · 어두운 사진이 반씩
            log_r = 0.03 * rng.normal() + (0.12 if lum < T.DARK else 0.0)                     # 어두우면 스케일이 더 틀린다
            rows.append(dict(split="train" if i < 30 else "val", seq="_s1" if i % 4 < 2 else "_s2", name=f"{i}.png", valid=1000, lum=lum, ud_absrel=0.1, ud_d1=0.9, log_r=log_r,
                             ud_med_absrel=0.08, da_shape_absrel=0.06, ud_shape_absrel=0.05, da_shape_d1=0.95, ud_shape_d1=0.96, flip_ud=abs(log_r) + 0.002 * rng.random(), disagree=0.05,
                             ud_conf=1.0, fx_rel=0.6))
            c = rng.random((200, len(T.CUES)))
            err = 0.02 + 0.2 * c[:, 1] + 0.01 * rng.random(200) + (0.02 if lum < T.DARK else 0.0)  # 사진 안에서는 둘째 단서 (flip_da) 를 따라가고, 어두운 사진은 통째로 조금 더 틀린다
            c[:, 0] = lum                                                                     # 밝기는 사진 사이에서만 다르다 → 사진 안의 순위에는 쓸모없다
            pxs.append(np.concatenate([c, err[:, None], err[:, None]], 1))
            img.append(np.full(200, i))
        old, T.MIN_GROUP = T.MIN_GROUP, 10
        try:
            lines, s = T.tables(rows, np.concatenate(pxs), np.concatenate(img))
        finally:
            T.MIN_GROUP = old
        self.assertTrue(all(l.startswith(("[표1]", "[표2]", "[표3]", "[판정]")) for l in lines))
        self.assertLessEqual(len(lines), 40)                                                   # 화면 끝 50 줄 안에 들어와야 한다
        self.assertTrue(all(len(l) < 400 for l in lines))
        v = s["verdict"]
        self.assertEqual(v["judged_groups"], ["밝음", "어두움"])
        self.assertGreater(s["pixel"]["flip_da/da/전체"]["rho"], 0.9)
        self.assertGreater(min(s["pixel"][f"모두 (릿지)/da/{g}"]["rho"] for g in ("밝음", "어두움")), 0.9)
        self.assertLess(abs(s["pixel"]["lum/da/전체"]["rho"]) if np.isfinite(s["pixel"]["lum/da/전체"]["rho"]) else 0.0, 0.2)
        self.assertEqual(v["pixel_conf"], "쓸 만함")
        self.assertEqual(v["shape"], "UniDepthV2")                                            # 두 묶음 모두 0.05 대 0.06 — 기준 0.005 를 넘는 차이
        self.assertEqual(v["scale"], "신뢰도와 함께만")                                         # 밝음은 0.02 쯤 (쓸 만함), 어두움은 0.12 쯤 → 나쁜 쪽을 따른다
        self.assertGreater(s["val 어두움"]["scale_bias"], 0.08)                                # 어두운 쪽은 한쪽으로 치우친 오차
        self.assertLess(s["val 어두움"]["scale_spread"], 0.05)
        self.assertGreater(s["image"]["flip_ud/밝음"]["rho"], 0.8)                             # 같은 밝기끼리도 순위를 맞추는 단서
        self.assertIn(v["image_conf"], ("쓸 만함", "못 씀"))
        self.assertEqual(set(v), {"judged_groups", "scale", "shape", "pixel_conf", "image_conf"})

    def test_tables_small(self):
        import teacher_check as T
        rng = np.random.default_rng(2)
        rows = [dict(split="train" if i < 8 else "val", seq="_s", name=f"{i}.png", valid=600, lum=0.4, ud_absrel=0.1, ud_d1=0.9, log_r=0.05 * rng.normal(), ud_med_absrel=0.08,
                     da_shape_absrel=0.06, ud_shape_absrel=0.06, da_shape_d1=0.9, ud_shape_d1=0.9, flip_ud=rng.random(), disagree=rng.random(), ud_conf=rng.random(), fx_rel=0.6) for i in range(16)]
        px = rng.random((16 * 60, len(T.CUES) + 2))
        lines, s = T.tables(rows, px, np.repeat(np.arange(16), 60))                            # smoke 묶음 (16 장) 에서도 끝까지 간다 — 묶음이 작으면 전체로 판정
        self.assertEqual(s["verdict"]["judged_groups"], ["전체"])
        self.assertEqual(s["verdict"]["shape"], "차이 없음 (Depth Anything V2 유지)")
        self.assertTrue(any(l.startswith("[판정] 영상 신뢰도") for l in lines))


if __name__ == "__main__":
    unittest.main(verbosity=1)
