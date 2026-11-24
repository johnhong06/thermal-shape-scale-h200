"""데이터 준비: 묶음 (tar) 들을 찾거나 받아서 sha256 을 확인하고 <dest> 에 푼다. 목록 = setup/data.json.

묶음은 이 저장소에 없고 받는 주소도 저장소에 적지 않는다 (다시 배포하면 안 되는 데이터). 파일마다 찾는 순서:
  1. --search 폴더들 (H200 = /app/data, 로컬 시험 = DATA_DIR) 아래 5 단계 안에서 크기가 꼭 맞는 파일 — 올려 둔 사본 (이름 · 폴더는 상관없다)
  2. 없으면 --links 로 받은 목록의 Google Drive 파일에서 받는다 (이어받기, 3 번까지)
     큰 파일에 Drive 가 확인 페이지 ('바이러스 검사를 못 함') 를 주면 그 양식대로 한 번 더 요청한다 (브라우저의 '무시하고 다운로드').
     그래도 웹 페이지가 오면 제목을 찍고 5 분 · 10 분 기다렸다가 다시 받아 본다 — 작업 하나에서 모두 합쳐 두 번까지.
     받기 한도 ('Quota exceeded') 는 기다려도 곧 풀리지 않으므로 (길면 하루) 바로 끝낸다. 받기 전에 받을 묶음을 모두 먼저 물어보고,
     하나라도 한도에 걸려 있으면 아무것도 받지 않는다 (몇 개만 받고 실패하면 한도만 더 쓴다). 한도를 돌아가지는 않는다.
     한도는 파일 전체를 요청할 때만 보인다 — 첫 바이트만 묻는 요청은 한도에 걸려 있어도 통과한다 (2026-11-24 에 확인).
     --links = 작은 목록 파일 (JSON: {묶음 이름: Drive 파일 ID}) 의 Drive 파일 ID. run.sh 의 LINKS= 인자로 온다 (실행 명령에만 적는 값).
     환경 변수 TD_HF_REPO (<계정>/<저장소>) · TD_HF_TOKEN (읽기 전용 토큰) 이 있으면 Drive 대신 Hugging Face 의 **비공개** 데이터 저장소에서 받는다
     (run.sh 의 DATA_REPO= · DATA_TOKEN=, 실행 명령에만 적는 값). 파일 이름 = 묶음 이름. 받기 전에 모든 묶음의 크기를 먼저 물어 본다.
     토큰은 huggingface.co 에만 보낸다 (다른 호스트로 넘겨질 때는 떼어 낸다). 저장소 이름 · 토큰은 화면에 찍지 않는다.
어느 쪽이든 sha256 이 data.json 과 같아야 쓴다. 풀 때는 이름이 정해진 꼴인 PNG 만 (정규 파일만, 경로는 우리가 다시 만든다).
끝나면 PNG 개수를 확인하고 <dest>/.done 에 '<종류> <해시>' 를 남긴다 (다시 부르면 건너뜀). 받은 tar 는 풀고 나서 지운다.
화면에 주소 · 파일 ID 를 찍지 않는다. 표준 라이브러리만 쓴다.

  --only <묶음 이름,…> 를 주면 종류와 상관없이 그 묶음들만 (상대 깊이 모델 점검: val.tar · ms2_rgb_val.tar).
사용: python3 setup/get_data.py --kind smoke|all|teacher|teacher_smoke [--only <묶음 이름,…>] --dest <폴더> --dl <받을 폴더> [--search /app/data …] [--links <Drive 파일 ID>]
"""
import argparse
import hashlib
import html
import json
import os
import re
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = json.load(open(os.path.join(HERE, "data.json"), encoding="utf-8"))
NAME = re.compile(r"^(?:(?:sync_data|proj_depth)/_[0-9-]+/(?:thr|rgb)/(?:img_left|depth_filtered)/[0-9A-Za-z_-]+\.png"
                  r"|firestereo/[a-z0-9_]+_test/(?:img_left|depth_filtered)/[0-9]+\.png"
                  r"|vivid/outdoor_robust_(?:night|day)[12]/(?:Thermal|Depth_T)/[0-9]+\.png"
                  r"|vivid/outdoor_robust_day[12]/RGB/[0-9]+\.png"
                  r"|vivid/indoor_[a-z_]+/(?:Thermal|Depth_T)/[0-9]+\.png"
                  r"|sthereo/sthereo_[0-9]{2}_[a-z_]+/(?:Thermal|Depth_T)/[0-9]{19}\.png"
                  r"|pairs/kaist/set[0-9]{2}/V[0-9]{3}/(?:lwir|visible)/I[0-9]{5}\.jpg"
                  r"|pairs/llvip/(?:infrared|visible)/[0-9]{6}\.jpg"
                  r"|pairs/freiburg/seq_[0-9]{2}_(?:day|night)/[0-9]{2}/(?:ir/[0-9]+_[0-9]+\.png|rgb/[0-9]+_[0-9]+\.jpg)"
                  r"|pairs/tartan/day[1245]/[0-9]{8}_[0-9]{6}/(?:thermal/[0-9]{8}\.png|rgb/[0-9]{8}\.jpg))$")


ID = re.compile(r"^[A-Za-z0-9_-]{20,80}$")


def drive_url(fid):
    return f"https://drive.usercontent.google.com/download?id={fid}&export=download&confirm=t"


UA = "thermal-h200-setup"
FORM_HOSTS = ("https://drive.usercontent.google.com/", "https://drive.google.com/")   # 확인 양식을 낼 곳은 Drive 뿐
WAITS = [300, 600]                                                # 웹 페이지가 오면 기다렸다 다시 받아 보는 시간 (초) — 작업 하나에서 모두 합쳐 이만큼만


def page(r):
    """Drive 가 파일 대신 준 웹 페이지 → (제목, 확인 양식대로 만든 주소 또는 None).
    제목은 화면에 찍으므로 ID 꼴의 긴 토큰을 가린다. 양식 = <form id="download-form"> 의 action + 숨은 칸 (id · export · confirm · uuid)."""
    try:
        t = r.read(1 << 18).decode("utf-8", "replace")
    except Exception:
        return "읽지 못함", None
    m = re.search(r"<title[^>]*>(.*?)</title>", t, re.S | re.I)
    title = re.sub(r"[A-Za-z0-9_-]{20,}", "…", re.sub(r"\s+", " ", html.unescape(m.group(1))).strip())[:60] if m else "제목 없음"
    f = re.search(r'<form[^>]*\bid="download-form"[^>]*>.*?</form>', t, re.S | re.I)
    if not f:
        return title, None
    act = re.search(r'<form[^>]*\baction="([^"]+)"', f.group(0), re.I)
    act = html.unescape(act.group(1)) if act else ""
    kv = [(html.unescape(k), html.unescape(v)) for k, v in
          re.findall(r'<input[^>]*\btype="hidden"[^>]*\bname="([^"]+)"[^>]*\bvalue="([^"]*)"', f.group(0), re.I)]
    if not (act.startswith(FORM_HOSTS) and kv):
        return title, None
    return title, act + ("&" if "?" in act else "?") + urllib.parse.urlencode(kv)


def request(fid, rng, timeout=120):
    """Drive 에 파일을 요청 → (열린 응답, None) 또는 (None, 웹 페이지의 제목). 확인 페이지가 오면 그 양식대로 한 번 더 요청한다."""
    url = drive_url(fid)
    for again in (True, False):
        r = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA, "Range": rng}), timeout=timeout)
        if "text/html" not in (r.headers.get("Content-Type") or ""):
            return r, None
        with r:
            title, form = page(r)
        if not (again and form and dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(form).query)).get("id") == fid):
            return None, title                                    # 양식이 없거나 (한도 등) 다른 파일을 가리키면 따라가지 않는다
        url = form
    return None, "?"


HF_BASE = "https://huggingface.co"
HF_REPO = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
HF_TOKEN = re.compile(r"^hf_[A-Za-z0-9]{20,100}$")
HF_WHY = {401: "토큰이 틀림 (401)", 403: "토큰에 이 저장소를 읽을 권한이 없음 (403)", 404: "저장소나 파일이 없음 — 또는 토큰에 권한이 없음 (404)"}


class DropAuth(urllib.request.HTTPRedirectHandler):
    """다른 호스트로 넘겨질 때 Authorization 을 떼어 낸다 — 토큰은 huggingface.co 에만 보낸다 (서명된 저장소 주소는 토큰이 붙으면 거절하기도 한다)."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urllib.parse.urlsplit(new.full_url).netloc != urllib.parse.urlsplit(req.full_url).netloc:
            for d in (new.headers, new.unredirected_hdrs):
                for k in [k for k in d if k.lower() == "authorization"]:
                    del d[k]
        return new


HF_OPEN = urllib.request.build_opener(DropAuth).open


def hf_env():
    """환경 변수의 (저장소, 토큰) · 없으면 None · 꼴이 틀리면 ValueError (값은 메시지에 넣지 않는다)."""
    repo, token = os.environ.get("TD_HF_REPO", ""), os.environ.get("TD_HF_TOKEN", "")
    if not (repo or token):
        return None
    if not (HF_REPO.match(repo) and HF_TOKEN.match(token)):
        raise ValueError("DATA_REPO (<계정>/<저장소>) · DATA_TOKEN (hf_…) 의 꼴이 맞지 않음 — 둘 다 줘야 한다")
    return repo, token


def hf_request(hf, name, rng, timeout=120):
    """Hugging Face 비공개 데이터 저장소의 파일을 요청 → (열린 응답, None) 또는 (None, 까닭). hf = (저장소, 토큰)."""
    url = f"{HF_BASE}/datasets/{hf[0]}/resolve/main/{urllib.parse.quote(name)}"
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Range": rng, "Authorization": f"Bearer {hf[1]}"})
    try:
        r = HF_OPEN(req, timeout=timeout)
    except urllib.error.HTTPError as ex:
        ex.close()
        if ex.code in HF_WHY:
            return None, HF_WHY[ex.code]
        raise
    if "text/html" in (r.headers.get("Content-Type") or ""):
        with r:
            return None, f"파일 대신 웹 페이지 '{page(r)[0]}'"
    return r, None


def hf_missing(hf, sizes):
    """받기 전 확인: 저장소에 없거나 · 못 읽거나 · 크기가 다른 묶음 [(이름, 까닭)]. 첫 바이트만 묻는다 (sizes = {이름: 크기})."""
    out = []
    for name, size in sizes.items():
        try:
            r, why = hf_request(hf, name, "bytes=0-0", timeout=60)
        except Exception as ex:
            out.append((name, f"접속 실패 ({type(ex).__name__})"))
            continue
        if r is None:
            out.append((name, why))
            continue
        with r:
            if not (r.headers.get("Content-Range") or "").endswith("/" + str(size)):
                out.append((name, "크기가 목록과 다름"))
    return out


def quota(title):
    return "quota" in (title or "").lower()


def blocked(links, names):
    """받기 한도에 걸린 묶음 이름들. 작업이 받는 꼴 (파일 전체) 로 요청하고 머리만 본 뒤 끊는다. 접속 문제 · 다른 웹 페이지는 받을 때 다시 본다."""
    out = []
    for name in names:
        try:
            r, title = request(links[name], "bytes=0-", timeout=60)
        except Exception:
            continue
        if r is not None:
            r.close()
        elif quota(title):
            out.append(name)
    return out


def read_links(fid):
    """목록 파일 (JSON, 수 KB) 을 받아 {묶음 이름: 파일 ID}. 주소 · ID 는 찍지 않는다."""
    if not ID.match(fid or ""):
        raise ValueError("LINKS 값의 꼴이 Drive 파일 ID 가 아님")
    req = urllib.request.Request(drive_url(fid), headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        if "text/html" in (r.headers.get("Content-Type") or ""):
            raise ValueError(f"목록 파일 대신 웹 페이지가 옴 '{page(r)[0]}' (공유가 꺼졌거나 ID 가 틀림)")
        d = json.loads(r.read(1 << 20).decode("utf-8"))
    if not isinstance(d, dict) or not all(isinstance(k, str) and isinstance(v, str) and ID.match(v) for k, v in d.items()):
        raise ValueError("목록 파일의 꼴이 다름")
    return d


def download(name, size, fid, dst, hf=None):
    """Google Drive (fid) 또는 Hugging Face (hf = (저장소, 토큰)) → dst (이어받기). 받은 크기가 맞으면 True."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    part = dst + ".part"
    t = 0
    while t < 3:
        have = os.path.getsize(part) if os.path.exists(part) else 0
        try:
            if hf:
                r, title = hf_request(hf, name, f"bytes={have}-")
                if r is None:                                     # 토큰 · 이름 문제는 다시 해도 같다
                    print(f"!!! [data] {name}: Hugging Face 에서 받지 못함 — {title}")
                    return False
            else:
                r, title = request(fid, f"bytes={have}-")
            if r is None:                                         # 파일 대신 웹 페이지 — 기다렸다 다시 (작업 하나에서 WAITS 만큼만)
                if quota(title):
                    print(f"!!! [data] {name}: Drive 받기 한도 ('{title}') — 기다려도 곧 풀리지 않는다 (길면 하루). 나중에 같은 명령을 다시")
                    return False
                if not WAITS:
                    print(f"!!! [data] {name}: Drive 가 파일 대신 웹 페이지를 줌 — '{title}' (받기 한도 · 공유 꺼짐)")
                    return False
                w = WAITS.pop(0)
                print(f"[data] {name}: Drive 가 파일 대신 웹 페이지를 줌 — '{title}' · {w // 60} 분 뒤 다시 받아 봄", flush=True)
                time.sleep(w)
                continue
            with r:
                mode = "ab" if r.status == 206 else "wb"
                got, mark = (have if mode == "ab" else 0), time.time()
                with open(part, mode) as fh:
                    while c := r.read(16 << 20):
                        fh.write(c)
                        got += len(c)
                        if time.time() - mark > 120:              # 2 분마다 한 줄 (멈춤 감지가 진행으로 보게)
                            print(f"[data] {name} 받는 중 {got / 2**30:.2f} / {size / 2**30:.2f} GiB", flush=True)
                            mark = time.time()
        except Exception as ex:
            t += 1
            print(f"[data] {name} 받기 끊김 ({t}/3, {type(ex).__name__}) — 30 초 뒤 이어받기", flush=True)
            time.sleep(30)
            continue
        t += 1
        if os.path.getsize(part) == size:
            os.replace(part, dst)
            return True
        if os.path.getsize(part) > size:
            os.remove(part)
    return False


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while c := fh.read(32 << 20):
            h.update(c)
    return h.hexdigest()


def search(roots, size, depth=5):
    for root in roots:
        if not os.path.isdir(root):
            continue
        base = root.rstrip("/").count("/")
        for d, subs, files in os.walk(root):
            if d.count("/") - base >= depth:
                subs[:] = []
            for f in files:
                p = os.path.join(d, f)
                try:
                    if os.path.getsize(p) == size:
                        yield p
                except OSError:
                    pass


def unpack(tar, dest):
    """이름이 정해진 꼴인 사진 (PNG · JPEG) 만 푼다 → 푼 개수."""
    n = 0
    with tarfile.open(tar, "r:") as t:
        for m in t:
            if not (m.isreg() and NAME.match(m.name)):
                continue
            p = os.path.join(dest, m.name)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with t.extractfile(m) as fi, open(p, "wb") as fo:
                shutil.copyfileobj(fi, fo, 4 << 20)
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["smoke", "all", "teacher", "teacher_smoke"], required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--dl", required=True)
    ap.add_argument("--search", nargs="*", default=[])
    ap.add_argument("--links", default="")
    ap.add_argument("--only", default="", help="묶음 이름 (쉼표) — 주면 --kind 대신 이것들만")
    a = ap.parse_args()
    only = [x for x in a.only.split(",") if x]
    if any(x not in SPEC["files"] for x in only):
        print(f"!!! [data] 목록 (data.json) 에 없는 묶음: {', '.join(x for x in only if x not in SPEC['files'])}")
        return 1
    files = {k: SPEC["files"][k] for k in only} if only else {k: v for k, v in SPEC["files"].items() if v["kind"] == a.kind}
    tag = ("only:" + ",".join(only) if only else a.kind) + " " + hashlib.sha256("".join(v["sha256"] for v in files.values()).encode()).hexdigest()[:16]
    done = os.path.join(a.dest, ".done")
    if os.path.exists(done) and open(done).read().strip() == tag:
        print("[data] 이미 풀려 있음 (같은 작업 안)")
        return 0
    try:
        hf = hf_env()
    except ValueError as ex:
        print(f"!!! [data] {ex}")
        return 1
    found = {name: next((c for c in search(a.search, f["size"]) if sha256(c) == f["sha256"]), None) for name, f in files.items()}
    lack = [n for n, p in found.items() if p is None]
    links = {}
    if lack and hf:                                                # Hugging Face 비공개 저장소 — 받기 전에 모두 있는지 · 크기가 맞는지
        bad = hf_missing(hf, {n: files[n]["size"] for n in lack})
        if bad:
            print(f"!!! [data] Hugging Face 저장소에서 받을 수 없는 묶음 {len(bad)} / {len(lack)} 개 — 아무것도 받지 않고 끝낸다")
            for why in dict.fromkeys(w for _, w in bad):
                print(f"!!! [data]   {why}: {', '.join(n for n, w in bad if w == why)}"[:390])
            return 1
    elif lack:                                                     # 올려 둔 사본이 없는 것만 받는다
        if not a.links:
            print(f"!!! [data] 묶음이 없음 ({len(lack)} / {len(files)}): {', '.join(lack)} — 실행 명령에 DATA_REPO= · DATA_TOKEN= 또는 LINKS= 를 주거나, 묶음을 /app/data 아래에 올린다")
            return 1
        try:
            links = read_links(a.links)
        except Exception as ex:
            print(f"!!! [data] 받을 곳 목록을 읽지 못함 ({type(ex).__name__}: {str(ex)[:120] if isinstance(ex, ValueError) else ''})")
            return 1
        no = [n for n in lack if n not in links]
        if no:
            print(f"!!! [data] 목록에 없는 묶음: {', '.join(no)}")
            return 1
        bl = blocked(links, lack)
        if bl:
            print(f"!!! [data] Drive 받기 한도 ('Quota exceeded') 에 걸린 묶음 {len(bl)} / {len(lack)} 개 — 아무것도 받지 않고 끝낸다 (길면 하루 뒤에 풀린다. 나중에 같은 명령을 다시)")
            print(f"!!! [data] 걸린 묶음: {', '.join(bl)}"[:390])
            return 1
    shutil.rmtree(a.dest, ignore_errors=True)
    os.makedirs(a.dest)
    t_all = time.time()
    for name, f in files.items():
        src, how, t0 = found[name], "올려 둔 사본", time.time()
        if src is None:
            src, how = os.path.join(a.dl, name), "Hugging Face" if hf else "Google Drive"
            if not download(name, f["size"], links.get(name), src, hf):
                print(f"!!! [data] {name} 을 받지 못함")
                return 1
            if sha256(src) != f["sha256"]:
                print(f"!!! [data] {name}: 받은 파일의 sha256 이 다름")
                return 1
        t1 = time.time()
        n = unpack(src, a.dest)
        ok = n == f["png"]
        print(f"[data] {name} {f['size'] / 2**30:.2f} GiB — {how} {t1 - t0:.0f} 초 · 풀기 {time.time() - t1:.0f} 초 · PNG {n:,} 개 " + ("(맞음)" if ok else f"!!! 기대 {f['png']:,}"), flush=True)
        if found[name] is None:                                    # 받은 tar 는 풀고 나서 지운다 (올려 둔 사본은 그대로)
            os.remove(src)
        if not ok:
            return 1
    with open(done, "w") as fh:
        fh.write(tag)
    print(f"[data] 끝 — 파일 {len(files)} 개 · {(time.time() - t_all) / 60:.1f} 분")
    return 0


if __name__ == "__main__":
    sys.exit(main())
