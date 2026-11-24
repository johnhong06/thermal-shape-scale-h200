"""setup/get_data.py 의 받기 — 이 PC 안의 작은 서버로 (표준 라이브러리만, 바깥 접속 없음).
Drive: 확인 페이지의 양식을 따라가는지 · 웹 페이지가 오면 제목을 찍고 정해진 만큼만 기다렸다 다시 받는지 · 한도면 바로 끝내는지 · 이어받기 · 화면에 ID 를 안 찍는지.
Hugging Face 비공개 저장소: 토큰을 그 호스트에만 보내는지 (다른 호스트로 넘겨질 때 떼어 냄) · 이어받기 · 토큰 · 이름이 틀릴 때 · 받기 전 확인 · 화면에 저장소 이름 · 토큰을 안 찍는지.
사용: python3 -I -B tests/test_get_data.py
"""
import contextlib
import http.server
import io
import os
import sys
import tempfile
import threading
import unittest
import urllib.parse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "setup"))
import get_data as G                                                                       # noqa: E402

FID = "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q"                                                  # Drive 파일 ID 꼴 (33 자, 지어낸 값)
DATA = bytes(range(256)) * 40
CONFIRM = ('<html><head><title>Google Drive - Virus scan warning</title></head><body>'
           '<form id="download-form" action="{base}/download" method="get"><input type="submit" id="uc-download-link" value="Download anyway"/>'
           '<input type="hidden" name="id" value="{fid}"><input type="hidden" name="export" value="download">'
           '<input type="hidden" name="confirm" value="t"><input type="hidden" name="uuid" value="0f1e2d3c-4b5a-6978-8695-a4b3c2d1e0f9"></form></body></html>')
QUOTA = "<html><head><title>Google Drive - Quota exceeded</title></head><body>Too many users have viewed or downloaded this file recently.</body></html>"
BUSY = "<html><head><title>Sorry, try again later {fid}</title></head><body>…</body></html>"     # 한도가 아닌 웹 페이지 (제목에 ID 가 섞인 경우)


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        sv = self.server
        q = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(self.path).query))
        sv.seen.append((q, self.headers.get("Range")))
        kind = sv.plan.pop(0) if sv.plan else "file"
        if kind in ("confirm", "quota", "busy", "other"):
            fid = "ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ" if kind == "other" else FID
            body = {"quota": QUOTA, "busy": BUSY}.get(kind, CONFIRM).format(base=sv.base, fid=fid).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
        else:
            start = int((self.headers.get("Range") or "bytes=0-")[6:].split("-")[0])
            body = DATA[start:]
            self.send_response(206)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Range", f"bytes {start}-{len(DATA) - 1}/{len(DATA)}")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class Download(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.sv.base = f"http://127.0.0.1:{cls.sv.server_address[1]}"
        cls.sv.plan, cls.sv.seen = [], []
        threading.Thread(target=cls.sv.serve_forever, daemon=True).start()
        cls.keep = (G.drive_url, G.FORM_HOSTS, list(G.WAITS))
        G.drive_url = lambda fid: f"{cls.sv.base}/download?id={fid}&export=download&confirm=t"
        G.FORM_HOSTS = (cls.sv.base + "/",)

    @classmethod
    def tearDownClass(cls):
        G.drive_url, G.FORM_HOSTS = cls.keep[:2]
        G.WAITS[:] = cls.keep[2]
        cls.sv.shutdown()
        cls.sv.server_close()

    def run_dl(self, plan, waits, have=0):
        self.sv.plan[:], self.sv.seen[:], G.WAITS[:] = plan, [], waits
        with tempfile.TemporaryDirectory() as t:
            dst = os.path.join(t, "dl", "x.tar")
            if have:
                os.makedirs(os.path.dirname(dst))
                with open(dst + ".part", "wb") as fh:
                    fh.write(DATA[:have])
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                ok = G.download("x.tar", len(DATA), FID, dst)
            got = None
            if os.path.exists(dst):
                with open(dst, "rb") as fh:
                    got = fh.read()
        self.assertNotIn(FID, out.getvalue())                                              # 화면에 ID 를 안 찍는다
        return ok, got, out.getvalue(), list(self.sv.seen)

    def test_plain_and_resume(self):
        ok, got, out, seen = self.run_dl([], [])
        self.assertTrue(ok and got == DATA and out == "" and len(seen) == 1)
        ok, got, out, seen = self.run_dl([], [], have=1000)
        self.assertTrue(ok and got == DATA)
        self.assertEqual(seen[0][1], "bytes=1000-")

    def test_confirm_form(self):
        ok, got, out, seen = self.run_dl(["confirm"], [])                                  # 확인 페이지 → 양식대로 한 번 더 → 파일. 기다리지 않는다
        self.assertTrue(ok and got == DATA and out == "")
        self.assertEqual(len(seen), 2)
        self.assertNotIn("uuid", seen[0][0])
        self.assertEqual((seen[1][0]["id"], seen[1][0]["confirm"], seen[1][0]["uuid"]), (FID, "t", "0f1e2d3c-4b5a-6978-8695-a4b3c2d1e0f9"))
        self.assertEqual(seen[1][1], "bytes=0-")                                           # 양식 요청에도 이어받기 범위를 준다

    def test_page_then_file(self):
        ok, got, out, seen = self.run_dl(["busy"], [0, 0])                                 # 웹 페이지 한 번 → 기다렸다 다시 → 파일
        self.assertTrue(ok and got == DATA)
        self.assertEqual(len(seen), 2)
        self.assertIn("'Sorry, try again later …'", out)                                   # 제목의 ID 꼴은 가린다
        self.assertIn("뒤 다시 받아 봄", out)
        self.assertEqual(G.WAITS, [0])                                                     # 기다린 만큼 줄어든다 (작업 하나에서 합쳐서 센다)

    def test_page_gives_up(self):
        ok, got, out, seen = self.run_dl(["busy"] * 9, [0, 0])                             # 두 번 기다리고 그만둔다
        self.assertFalse(ok)
        self.assertIsNone(got)
        self.assertEqual(len(seen), 3)
        self.assertIn("!!! [data] x.tar: Drive 가 파일 대신 웹 페이지를 줌 — 'Sorry, try again later …'", out)
        ok, got, out, seen = self.run_dl(["busy"] * 9, [])                                 # 남은 기다림이 없으면 바로
        self.assertFalse(ok)
        self.assertEqual(len(seen), 1)

    def test_quota_stops_at_once(self):
        ok, got, out, seen = self.run_dl(["quota"] * 9, [0, 0])                            # 받기 한도는 기다리지 않는다
        self.assertFalse(ok)
        self.assertEqual(len(seen), 1)
        self.assertEqual(G.WAITS, [0, 0])
        self.assertIn("!!! [data] x.tar: Drive 받기 한도 ('Google Drive - Quota exceeded')", out)

    def test_blocked(self):
        self.sv.plan[:], self.sv.seen[:] = ["file", "quota", "busy", "confirm", "file", "confirm", "quota"], []
        names = ["a.tar", "b.tar", "c.tar", "d.tar", "e.tar"]                              # 파일 · 한도 · 다른 페이지 · 확인 뒤 파일 · 확인 뒤 한도
        self.assertEqual(G.blocked({n: FID for n in names}, names), ["b.tar", "e.tar"])
        self.assertEqual([x[1] for x in self.sv.seen], ["bytes=0-"] * 7)                   # 작업이 받는 꼴로 묻는다 (첫 바이트만 물으면 한도가 안 보인다)

    def test_form_not_followed(self):
        ok, got, out, seen = self.run_dl(["other"] + ["busy"] * 9, [])                     # 양식이 다른 파일을 가리키면 따라가지 않는다
        self.assertFalse(ok)
        self.assertEqual(len(seen), 1)
        ok, got, out, seen = self.run_dl(["confirm", "confirm"] + ["busy"] * 9, [])        # 양식은 한 번만 따라간다
        self.assertFalse(ok)
        self.assertEqual(len(seen), 2)
        self.assertIn("Virus scan warning", out)

    def test_page_parse(self):
        base = self.sv.base
        t, u = G.page(io.BytesIO(CONFIRM.format(base=base, fid=FID).encode()))
        self.assertEqual(t, "Google Drive - Virus scan warning")
        self.assertEqual(u, f"{base}/download?id={FID}&export=download&confirm=t&uuid=0f1e2d3c-4b5a-6978-8695-a4b3c2d1e0f9")
        self.assertEqual(G.page(io.BytesIO(CONFIRM.format(base="https://example.com", fid=FID).encode()))[1], None)   # Drive 가 아닌 곳의 양식
        self.assertEqual(G.page(io.BytesIO(QUOTA.encode())), ("Google Drive - Quota exceeded", None))
        self.assertEqual(G.page(io.BytesIO(BUSY.format(fid=FID).encode())), ("Sorry, try again later …", None))
        self.assertTrue(G.quota("Google Drive - Quota exceeded") and not G.quota("Google Drive - Virus scan warning") and not G.quota(None))
        self.assertEqual(G.page(io.BytesIO(b"<html>no title</html>")), ("제목 없음", None))
        self.assertLessEqual(len(G.page(io.BytesIO(b"<title>" + b"ab " * 200 + b"</title>"))[0]), 60)


REPO, TOKEN = "some-user/some-data", "hf_" + "Ab1" * 11                                    # 지어낸 값


class HubHandler(http.server.BaseHTTPRequestHandler):
    """가짜 huggingface.co: 토큰을 확인하고 다른 호스트 (CDN) 의 서명된 주소로 넘긴다. rel.tar 는 같은 호스트 안에서 넘긴다."""

    def log_message(self, *a):
        pass

    def do_GET(self):
        sv = self.server
        auth = self.headers.get("Authorization")
        sv.seen.append((self.path, auth, self.headers.get("Range")))
        pre = f"/datasets/{REPO}/resolve/main/"
        code, loc = 404, None
        if auth != f"Bearer {TOKEN}":
            code = 401
        elif self.path == "/same/rel.tar":
            code, loc = 302, f"{sv.cdn}/blob/rel.tar?sig=1"
        elif self.path.startswith(pre) and self.path[len(pre):] in sv.names:
            name = self.path[len(pre):]
            code, loc = 302, ("/same/rel.tar" if name == "rel.tar" else f"{sv.cdn}/blob/{name}?sig=1")
        self.send_response(code)
        if loc:
            self.send_header("Location", loc)
        self.send_header("Content-Length", "0")
        self.end_headers()


class CdnHandler(http.server.BaseHTTPRequestHandler):
    """가짜 저장소 호스트: 서명된 주소에 토큰이 붙어 오면 거절한다."""

    def log_message(self, *a):
        pass

    def do_GET(self):
        sv = self.server
        sv.seen.append((self.path, self.headers.get("Authorization"), self.headers.get("Range")))
        if self.headers.get("Authorization"):
            self.send_response(400)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        rng = self.headers.get("Range") or "bytes=0-"
        a, b = rng[6:].split("-")
        a = int(a)
        b = int(b) if b else len(DATA) - 1
        body = DATA[a:b + 1]
        self.send_response(206)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Range", f"bytes {a}-{b}/{len(DATA)}")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class Hub(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hub = http.server.ThreadingHTTPServer(("127.0.0.1", 0), HubHandler)
        cls.cdn = http.server.ThreadingHTTPServer(("127.0.0.1", 0), CdnHandler)
        cls.hub.cdn = f"http://127.0.0.1:{cls.cdn.server_address[1]}"
        cls.hub.names, cls.hub.seen, cls.cdn.seen = {"x.tar", "y.tar", "rel.tar"}, [], []
        for sv in (cls.hub, cls.cdn):
            threading.Thread(target=sv.serve_forever, daemon=True).start()
        cls.keep = G.HF_BASE
        G.HF_BASE = f"http://127.0.0.1:{cls.hub.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        G.HF_BASE = cls.keep
        for sv in (cls.hub, cls.cdn):
            sv.shutdown()
            sv.server_close()

    def run_dl(self, name, hf, have=0):
        self.hub.seen[:], self.cdn.seen[:] = [], []
        with tempfile.TemporaryDirectory() as t:
            dst = os.path.join(t, "dl", name)
            if have:
                os.makedirs(os.path.dirname(dst))
                with open(dst + ".part", "wb") as fh:
                    fh.write(DATA[:have])
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                ok = G.download(name, len(DATA), None, dst, hf)
            got = None
            if os.path.exists(dst):
                with open(dst, "rb") as fh:
                    got = fh.read()
        for secret in (TOKEN, REPO, hf[1]):                                                 # 화면에 저장소 이름 · 토큰을 안 찍는다
            self.assertNotIn(secret, out.getvalue())
        return ok, got, out.getvalue()

    def test_download_and_resume(self):
        ok, got, out = self.run_dl("x.tar", (REPO, TOKEN))
        self.assertTrue(ok and got == DATA and out == "")
        self.assertEqual(self.hub.seen, [(f"/datasets/{REPO}/resolve/main/x.tar", f"Bearer {TOKEN}", "bytes=0-")])
        self.assertEqual(self.cdn.seen, [("/blob/x.tar?sig=1", None, "bytes=0-")])         # 다른 호스트에는 토큰을 안 보낸다 · 범위는 그대로 간다
        ok, got, out = self.run_dl("x.tar", (REPO, TOKEN), have=1000)
        self.assertTrue(ok and got == DATA)
        self.assertEqual(self.cdn.seen[0][2], "bytes=1000-")

    def test_same_host_redirect_keeps_token(self):
        ok, got, out = self.run_dl("rel.tar", (REPO, TOKEN))                                # 같은 호스트 안에서 넘겨질 때는 토큰을 그대로
        self.assertTrue(ok and got == DATA)
        self.assertEqual([x[1] for x in self.hub.seen], [f"Bearer {TOKEN}"] * 2)
        self.assertEqual(self.cdn.seen[0][1], None)

    def test_bad_token_or_name(self):
        ok, got, out = self.run_dl("x.tar", (REPO, "hf_" + "Zz9" * 11))                     # 틀린 토큰 — 다시 해 보지 않고 바로 끝
        self.assertFalse(ok)
        self.assertEqual(len(self.hub.seen), 1)
        self.assertIn("!!! [data] x.tar: Hugging Face 에서 받지 못함 — 토큰이 틀림 (401)", out)
        ok, got, out = self.run_dl("nope.tar", (REPO, TOKEN))
        self.assertFalse(ok)
        self.assertIn("(404)", out)

    def test_missing(self):
        bad = G.hf_missing((REPO, TOKEN), {"x.tar": len(DATA), "y.tar": len(DATA) + 1, "nope.tar": 5})
        self.assertEqual([n for n, _ in bad], ["y.tar", "nope.tar"])
        self.assertEqual(bad[0][1], "크기가 목록과 다름")
        self.assertIn("(404)", bad[1][1])
        self.assertEqual(G.hf_missing((REPO, TOKEN), {"x.tar": len(DATA)}), [])
        self.assertEqual([w for _, w in G.hf_missing((REPO, "hf_" + "Zz9" * 11), {"x.tar": len(DATA)})], ["토큰이 틀림 (401)"])

    def test_env(self):
        keep = {k: os.environ.pop(k, None) for k in ("TD_HF_REPO", "TD_HF_TOKEN")}
        try:
            self.assertIsNone(G.hf_env())
            os.environ.update(TD_HF_REPO=REPO, TD_HF_TOKEN=TOKEN)
            self.assertEqual(G.hf_env(), (REPO, TOKEN))
            for repo, token in ((REPO, ""), ("", TOKEN), ("no-slash", TOKEN), (REPO, "token"), (REPO + "/x", TOKEN), ("../x/y", TOKEN)):
                os.environ.update(TD_HF_REPO=repo, TD_HF_TOKEN=token)
                with self.assertRaises(ValueError) as cm:
                    G.hf_env()
                self.assertNotIn(TOKEN, str(cm.exception))                                 # 값은 메시지에 안 넣는다
                self.assertNotIn(REPO, str(cm.exception))
        finally:
            for k, v in keep.items():
                os.environ.pop(k, None)
                if v is not None:
                    os.environ[k] = v


if __name__ == "__main__":
    unittest.main(verbosity=1)
