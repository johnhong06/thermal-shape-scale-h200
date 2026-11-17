"""부품 시험 (CPU, 가짜 데이터) — 지표 · 손실 · 톤매핑 · 데이터 순서 · 모델 입출력 · 평가 · 표 · 묶음 풀기 · 선생 점검의 계산.

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
    def put(img, dep, n, h):
        os.makedirs(img, exist_ok=True)
        os.makedirs(dep, exist_ok=True)
        for i in n:
            Image.fromarray(rng.integers(3000, 6000, (h, 640), dtype=np.uint16)).save(os.path.join(img, f"{i:06d}.png"))
            d = (rng.uniform(2, 60, (h, 640)) * 256).astype(np.uint16)
            d[rng.random((h, 640)) < 0.7] = 0                       # 정답이 드문 LiDAR 처럼
            Image.fromarray(d).save(os.path.join(dep, f"{i:06d}.png"))
    for name in D.MS2_SETS:
        seqs = [s.strip() for s in open(os.path.join(HERE, "..", "splits", "ms2", f"{name}_list.txt")) if s.strip()]
        for s in seqs[:2 if name == "train" else 1]:
            put(os.path.join(root, "sync_data", s, "thr", "img_left"), os.path.join(root, "proj_depth", s, "thr", "depth_filtered"), (7, 3, 5), 256)
    for d in ("hawkins_9_test", "frick_9_test"):
        put(os.path.join(root, "firestereo", d, "img_left"), os.path.join(root, "firestereo", d, "depth_filtered"), (2, 1), 512)
    for s in ("outdoor_robust_night1", "outdoor_robust_night2", "outdoor_robust_day1"):
        put(os.path.join(root, "vivid", s, "Thermal"), os.path.join(root, "vivid", s, "Depth_T"), (4, 2), 512)
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

    def test_pairs_extra(self):
        f, t = D.Pairs(["freiburg"]), D.Pairs(["tartan"])
        self.assertEqual((len(f), len(t)), (1, 1))
        x, r = f[0]
        self.assertEqual((tuple(x.shape), tuple(r.shape), x.dtype, r.dtype), ((3, 298, 640), (3, 298, 640), torch.float32, torch.float32))
        self.assertTrue(torch.equal(x[0] * 0.229 + 0.485, x[1] * 0.224 + 0.456) or torch.allclose(x[0] * 0.229 + 0.485, x[1] * 0.224 + 0.456, atol=1e-5))   # 열화상은 세 채널이 같은 값
        x, r = t[0]
        self.assertEqual((tuple(x.shape), tuple(r.shape)), ((3, 512, 640), (3, 512, 640)))
        self.assertTrue(torch.isfinite(x).all() and torch.isfinite(r).all())

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

    def test_report(self):
        import contextlib
        import report as R

        def met(a):
            return {v: {k: a for k in C.METRICS} for v in C.VARIANTS} | {"n": 1, "n_all": 1, "scale": {"abs_med": a / 2, "bias": -a / 2}}
        with tempfile.TemporaryDirectory() as t:
            for arm, fire in (("base", (0.300, 0.310, 0.305)), ("plv", (0.250, 0.262, 0.255))):
                for s, f in enumerate(fire):
                    d = os.path.join(t, "runs", f"{arm}_s{s}", "eval")
                    os.makedirs(d)
                    m = {k: met(0.075 + 0.001 * s) for k in ("test_day", "test_night", "test_rainy")} | {"fire": met(f) | {"day": met(f), "night": met(f)}}
                    m |= {"vivid": met(0.160 + 0.004 * s), "vivid_day": met(0.150)}
                    json.dump(m, open(os.path.join(d, "metrics.json"), "w"))
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                R.summary(os.path.join(t, "runs"), os.path.join(t, "summary"))
            out = buf.getvalue().splitlines()
            self.assertLessEqual(len(out), 40)                     # run.sh 가 남기는 끝 50 줄 안에 들어가야 한다
            self.assertTrue(all(len(x) < 400 for x in out))
            j = json.load(open(os.path.join(t, "summary", "summary.json")))["judge"]
            self.assertTrue(j["gate"]["ok"] and j["gap"]["FIReStereo"]["ok"] and j["gap"]["ViViD 밤"]["ok"])
            self.assertTrue(j["plv"]["FIReStereo"]["verdict"].startswith("개선"))  # −0.049 < −잡음 폭 0.010, MS2 는 그대로
            self.assertAlmostEqual(j["plv"]["FIReStereo"]["d_zs"], -0.049333, places=5)
            self.assertTrue(j["plv"]["ViViD 밤"]["verdict"].startswith("차이 없음"))  # 같은 값 → 잡음 폭 0.008 안
            self.assertTrue(any(x.startswith("[표5] base") for x in out))
            self.assertTrue(any(x.startswith("[표6] base  | 0.0380") for x in out))   # MS2 평균 (0.075 + 0.076 + 0.077) / 3 / 2
            os.remove(os.path.join(t, "runs", "plv_s1", "eval", "metrics.json"))
            os.remove(os.path.join(t, "runs", "plv_s2", "eval", "metrics.json"))
            with contextlib.redirect_stdout(io.StringIO()):
                R.summary(os.path.join(t, "runs"), os.path.join(t, "summary"))
            self.assertIn("시드가 하나", json.load(open(os.path.join(t, "summary", "summary.json")))["judge"]["plv"]["FIReStereo"]["verdict"])
            # 이 작업에 base 가 없으면 앞선 작업의 base (setup/base_ref.json 꼴: 시드별 raw AbsRel 만) 를 기준으로 판정한다
            import shutil
            for s in range(3):
                shutil.rmtree(os.path.join(t, "runs", f"base_s{s}"))
                d = os.path.join(t, "runs", f"pl_s{s}", "eval")
                os.makedirs(d)
                m = {k: met(0.075) for k in ("test_day", "test_night", "test_rainy")} | {"fire": met(0.20 + 0.002 * s), "vivid": met(0.12)}
                json.dump(m, open(os.path.join(d, "metrics.json"), "w"))
            ref = {"source": "앞선 작업", "seeds": {str(s): {k: {"raw": {"abs_rel": v}} for k, v in
                   (("test_day", 0.075), ("test_night", 0.075 + 0.001 * s), ("test_rainy", 0.075), ("fire", 0.30 + 0.005 * s), ("vivid", 0.16 + 0.004 * s))} for s in range(3)}}
            old, R.BASE_REF = R.BASE_REF, os.path.join(t, "base_ref.json")
            json.dump(ref, open(R.BASE_REF, "w"))
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                R.summary(os.path.join(t, "runs"), os.path.join(t, "summary"))
            R.BASE_REF = old
            j = json.load(open(os.path.join(t, "summary", "summary.json")))["judge"]
            self.assertTrue(j["pl"]["FIReStereo"]["verdict"].startswith("개선") and j["pl"]["ViViD 밤"]["verdict"].startswith("개선"))
            self.assertIn("(앞선 작업)", buf.getvalue())
            self.assertLessEqual(len(buf.getvalue().splitlines()), 40)
            for arm in ("base", "plv"):                            # 팔 셋 × 시드 셋 (한 작업에서 전부 돌릴 때) 도 끝 50 줄 안에 든다 (zip · 끝 줄 2 개 포함)
                for s in range(3):
                    d = os.path.join(t, "runs", f"{arm}_s{s}", "eval")
                    os.makedirs(d, exist_ok=True)
                    json.dump({k: met(0.075) for k in ("test_day", "test_night", "test_rainy")} | {"fire": met(0.3) | {"day": met(0.3), "night": met(0.3)}, "vivid": met(0.16), "vivid_day": met(0.15)},
                              open(os.path.join(d, "metrics.json"), "w"))
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                R.summary(os.path.join(t, "runs"), os.path.join(t, "summary"))
            self.assertLessEqual(len(buf.getvalue().splitlines()), 46)
            self.assertTrue(all(len(x) < 400 for x in buf.getvalue().splitlines()))

    def test_unpack(self):
        import get_data as G
        with tempfile.TemporaryDirectory() as t:
            tar = os.path.join(t, "x.tar")
            good = ["sync_data/_2021-08-06-10-59-33/thr/img_left/000001.png", "proj_depth/_2021-08-06-10-59-33/thr/depth_filtered/000001.png",
                    "firestereo/frick_2_test/img_left/02250.png", "vivid/outdoor_robust_night1/Thermal/000001.png", "vivid/outdoor_robust_day2/Depth_T/000001.png",
                    "vivid/outdoor_robust_day1/RGB/000001.png", "pairs/kaist/set00/V000/lwir/I00010.jpg", "pairs/llvip/visible/010001.jpg",
                    "sync_data/_2021-08-06-10-59-33/rgb/img_left/000001.png", "proj_depth/_2021-08-06-10-59-33/rgb/depth_filtered/000001.png",   # RGB = 선생 점검용 묶음
                    "pairs/freiburg/seq_02_night/06/ir/1579632394_1905673020.png", "pairs/freiburg/seq_02_night/06/rgb/1579632394_1905673020.jpg",
                    "pairs/tartan/day4/20250828_000502/thermal/00000120.png", "pairs/tartan/day4/20250828_000502/rgb/00000120.jpg"]
            bad = ["../evil.png", "sync_data/_2021-08-06-10-59-33/calib.npy", "sync_data/../../x/thr/img_left/1.png", "firestereo/frick_2_train/img_left/1.png",
                   "sync_data/_2021-08-06-10-59-33/nir/img_left/000001.png", "/abs/sync_data/_1/thr/img_left/1.png", "vivid/indoor_robust_dark/Thermal/000001.png",
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
