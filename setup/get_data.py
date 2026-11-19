"""데이터 준비: 묶음 (tar) 들을 찾거나 받아서 sha256 을 확인하고 <dest> 에 푼다. 목록 = setup/data.json.

묶음은 이 저장소에 없고 받는 주소도 저장소에 적지 않는다 (다시 배포하면 안 되는 데이터). 파일마다 찾는 순서:
  1. --search 폴더들 (H200 = /app/data, 로컬 시험 = DATA_DIR) 아래 5 단계 안에서 크기가 꼭 맞는 파일 — 올려 둔 사본 (이름 · 폴더는 상관없다)
  2. 없으면 --links 로 받은 목록의 Google Drive 파일에서 받는다 (이어받기, 3 번까지)
     --links = 작은 목록 파일 (JSON: {묶음 이름: Drive 파일 ID}) 의 Drive 파일 ID. run.sh 의 LINKS= 인자로 온다 (실행 명령에만 적는 값).
어느 쪽이든 sha256 이 data.json 과 같아야 쓴다. 풀 때는 이름이 정해진 꼴인 PNG 만 (정규 파일만, 경로는 우리가 다시 만든다).
끝나면 PNG 개수를 확인하고 <dest>/.done 에 '<종류> <해시>' 를 남긴다 (다시 부르면 건너뜀). 받은 tar 는 풀고 나서 지운다.
화면에 주소 · 파일 ID 를 찍지 않는다. 표준 라이브러리만 쓴다.

  --only <묶음 이름,…> 를 주면 종류와 상관없이 그 묶음들만 (상대 깊이 모델 점검: val.tar · ms2_rgb_val.tar).
사용: python3 setup/get_data.py --kind smoke|all|teacher|teacher_smoke [--only <묶음 이름,…>] --dest <폴더> --dl <받을 폴더> [--search /app/data …] [--links <Drive 파일 ID>]
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tarfile
import time
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


def read_links(fid):
    """목록 파일 (JSON, 수 KB) 을 받아 {묶음 이름: 파일 ID}. 주소 · ID 는 찍지 않는다."""
    if not ID.match(fid or ""):
        raise ValueError("LINKS 값의 꼴이 Drive 파일 ID 가 아님")
    req = urllib.request.Request(drive_url(fid), headers={"User-Agent": "thermal-h200-setup"})
    with urllib.request.urlopen(req, timeout=60) as r:
        if "text/html" in (r.headers.get("Content-Type") or ""):
            raise ValueError("목록 파일 대신 웹 페이지가 옴 (공유가 꺼졌거나 ID 가 틀림)")
        d = json.loads(r.read(1 << 20).decode("utf-8"))
    if not isinstance(d, dict) or not all(isinstance(k, str) and isinstance(v, str) and ID.match(v) for k, v in d.items()):
        raise ValueError("목록 파일의 꼴이 다름")
    return d


def download(name, size, fid, dst):
    """Google Drive → dst (이어받기). 받은 크기가 맞으면 True."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    part = dst + ".part"
    for t in range(3):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        req = urllib.request.Request(drive_url(fid), headers={"User-Agent": "thermal-h200-setup", "Range": f"bytes={have}-"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                if "text/html" in (r.headers.get("Content-Type") or ""):
                    print(f"!!! [data] {name}: Drive 가 파일 대신 웹 페이지를 줌 (받기 한도 · 공유 꺼짐)")
                    return False
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
            print(f"[data] {name} 받기 끊김 ({t + 1}/3, {type(ex).__name__}) — 30 초 뒤 이어받기", flush=True)
            time.sleep(30)
            continue
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
    found = {name: next((c for c in search(a.search, f["size"]) if sha256(c) == f["sha256"]), None) for name, f in files.items()}
    lack = [n for n, p in found.items() if p is None]
    links = {}
    if lack:                                                       # 올려 둔 사본이 없는 것만 받는다
        if not a.links:
            print(f"!!! [data] 묶음이 없음 ({len(lack)} / {len(files)}): {', '.join(lack)} — 실행 명령에 LINKS=<목록 파일 ID> 를 주거나, 묶음을 /app/data 아래에 올린다 (README '데이터')")
            return 1
        try:
            links = read_links(a.links)
        except Exception as ex:
            print(f"!!! [data] 받을 곳 목록을 읽지 못함 ({type(ex).__name__}: {str(ex)[:80] if isinstance(ex, ValueError) else ''})")
            return 1
        no = [n for n in lack if n not in links]
        if no:
            print(f"!!! [data] 목록에 없는 묶음: {', '.join(no)}")
            return 1
    shutil.rmtree(a.dest, ignore_errors=True)
    os.makedirs(a.dest)
    t_all = time.time()
    for name, f in files.items():
        src, how, t0 = found[name], "올려 둔 사본", time.time()
        if src is None:
            src, how = os.path.join(a.dl, name), "Google Drive"
            if not download(name, f["size"], links[name], src):
                print(f"!!! [data] {name} 을 받지 못함")
                return 1
            if sha256(src) != f["sha256"]:
                print(f"!!! [data] {name}: 받은 파일의 sha256 이 다름")
                return 1
        t1 = time.time()
        n = unpack(src, a.dest)
        ok = n == f["png"]
        print(f"[data] {name} {f['size'] / 2**30:.2f} GiB — {how} {t1 - t0:.0f} 초 · 풀기 {time.time() - t1:.0f} 초 · PNG {n:,} 개 " + ("(맞음)" if ok else f"!!! 기대 {f['png']:,}"), flush=True)
        if how == "Google Drive":
            os.remove(src)
        if not ok:
            return 1
    with open(done, "w") as fh:
        fh.write(tag)
    print(f"[data] 끝 — 파일 {len(files)} 개 · {(time.time() - t_all) / 60:.1f} 분")
    return 0


if __name__ == "__main__":
    sys.exit(main())
