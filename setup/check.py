"""점검 — 아무것도 설치하지 않는다.

  python3 setup/check.py box  --app /app [--links <목록 파일의 Drive ID>]   # 컨테이너: 파이썬 · torch (있으면) · GPU 메모리 · CPU · RAM · /dev/shm · 경로 여유 · 쓰기 가능 ·
                                            #   /app/data 에 올려 둔 묶음 (크기로) · 외부 접속 (HF · codeload · PyPI · PyTorch 휠 · --links 를 주면 Drive 의 묶음마다)
  <학습 파이썬> setup/check.py env [--gpu]   # 학습 환경: 패키지 판 · CUDA · bf16 · 작은 GPU 연산
화면에는 이름과 숫자만 찍는다 (요청 이슈 댓글은 여럿이 본다) — Drive 주소 · 파일 ID 는 찍지 않는다.
"""
import argparse
import concurrent.futures as cf
import json
import os
import platform
import shutil
import sys
import tempfile
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = json.load(open(os.path.join(HERE, "data.json"), encoding="utf-8"))
W = {k: v for k, v in json.load(open(os.path.join(HERE, "weights.json"), encoding="utf-8")).items() if not k.startswith("_")}
COMMIT = "a561b849ebae10a6f5ef49e26c83cbbcd36c71bf"


def gib(n):
    return f"{n / 2**30:.1f} GiB"


def probe(name, url, headers=None, expect=None):
    req = urllib.request.Request(url, headers={"User-Agent": "thermal-h200-check", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return f"  접속 {name}: OK ({r.status}){expect(r) if expect else ''}"
    except Exception as ex:
        return f"  접속 {name}: 실패 ({type(ex).__name__})"


def probe_drive(name, fid, size):
    """작업이 받는 꼴 그대로 (파일 전체 · 확인 페이지가 오면 양식대로 한 번 더) 요청하고 머리만 본 뒤 끊는다.
    첫 바이트만 묻으면 받기 한도에 걸려 있어도 통과한다 (2026-11-24 에 확인) — 그래서 전체를 요청한다."""
    import get_data
    try:
        r, title = get_data.request(fid, "bytes=0-", timeout=20)
    except Exception as ex:
        return f"  접속 {name}: 실패 ({type(ex).__name__})"
    if r is None:
        why = "받기 한도 — 길면 하루 뒤에 풀린다" if get_data.quota(title) else "받기 막힘 · 공유 꺼짐"
        return f"  접속 {name}: !!! 파일이 아니라 웹 페이지 '{title}' ({why})"
    with r:
        cr = r.headers.get("Content-Range") or ""
        return f"  접속 {name}: OK ({r.status})" + (" — 크기 맞음" if cr.endswith("/" + str(size)) else " — 크기 확인 안 됨")


def box(app, links_id=""):
    print(f"[check] 파이썬 {platform.python_version()} · {platform.platform()}")
    try:
        import torch
        line = f"[check] torch {torch.__version__} · CUDA {torch.version.cuda} · GPU {torch.cuda.is_available()}"
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            line += f" · {torch.cuda.get_device_name(0)} · 메모리 {gib(free)} 비어 있음 / {gib(total)}"
        print(line)
    except Exception as ex:
        print(f"[check] 이미지 파이썬에 torch 없음 ({type(ex).__name__}) — 학습 환경은 setup 이 따로 만든다")
    try:
        ncpu = len(os.sched_getaffinity(0))
    except AttributeError:
        ncpu = os.cpu_count()
    quota = ""
    if os.path.exists("/sys/fs/cgroup/cpu.max"):
        q = open("/sys/fs/cgroup/cpu.max").read().split()
        quota = f" · cgroup {q[0]}/{q[1]}" if q and q[0] != "max" else " · cgroup 제한 없음"
    mem = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo") if l.split()[0] in ("MemTotal:", "MemAvailable:")}
    shm = shutil.disk_usage("/dev/shm").total if os.path.isdir("/dev/shm") else 0
    print(f"[check] CPU {ncpu}{quota} · RAM {gib(mem.get('MemAvailable', 0) * 1024)} 비어 있음 / {gib(mem.get('MemTotal', 0) * 1024)} · /dev/shm {gib(shm)}")
    for sub in ("data", "output", "scratch"):
        p = os.path.join(app, sub)
        if not os.path.isdir(p):
            print(f"[check] {p}: 없음")
            continue
        u = shutil.disk_usage(p)
        try:
            with tempfile.NamedTemporaryFile(dir=p):
                w = "쓰기 가능"
        except OSError:
            w = "읽기 전용"
        print(f"[check] {p}: 여유 {gib(u.free)} / {gib(u.total)} · {w}")
    sizes = {v["size"]: k for k, v in DATA["files"].items()}
    found, d0 = set(), os.path.join(app, "data")
    if os.path.isdir(d0):
        base = d0.count("/")
        for d, subs, files in os.walk(d0):
            if d.count("/") - base >= 5:
                subs[:] = []
            for f in files:
                try:
                    found.add(sizes.get(os.path.getsize(os.path.join(d, f))))
                except OSError:
                    pass
    found.discard(None)
    lack = sorted(set(sizes.values()) - found)
    print(f"[check] /app/data 에서 크기가 맞는 묶음: {len(found)} / {len(sizes)} 개" + (" — 없는 것은 LINKS= 로 준 곳에서 받는다" if lack else " (모두 있음)"))
    drive = []
    if links_id:
        sys.path.insert(0, HERE)
        import get_data
        try:
            links = get_data.read_links(links_id)
            print(f"[check] 받을 곳 목록: 읽음 — 묶음 {len(links)} 개" + ("" if set(DATA["files"]) <= set(links) else " (!!! 목록에 없는 묶음이 있음)"))
            drive = [(lambda k=k, v=v: probe_drive(f"Google Drive ({k}, {gib(v['size'])})", links[k], v["size"]))
                     for k, v in DATA["files"].items() if k in links]
        except Exception as ex:
            print(f"[check] 받을 곳 목록: !!! 읽지 못함 ({type(ex).__name__})")
    elif lack:
        print("[check] LINKS= 를 주지 않음 — Drive 에서 받을 수 있는지는 확인하지 않았다")
    m = W["dav2_vitl"]
    fn = next(iter(m["files"]))
    jobs = [
        ("Hugging Face API", f"https://huggingface.co/api/models/{m['repo']}/revision/{m['revision']}", None, None),
        ("Hugging Face 파일", f"https://huggingface.co/{m['repo']}/resolve/{m['revision']}/{fn}", {"Range": "bytes=0-0"}, None),
        ("codeload.github.com", f"https://codeload.github.com/DepthAnything/Depth-Anything-V2/tar.gz/{COMMIT}", {"Range": "bytes=0-0"}, None),
        ("PyPI", "https://pypi.org/simple/pillow/", None, None),
        ("PyTorch 휠 색인", "https://download.pytorch.org/whl/cu128/torch/", None, None),
    ]
    jobs = [(lambda j=j: probe(*j)) for j in jobs] + drive
    with cf.ThreadPoolExecutor(len(jobs)) as ex:
        for line in ex.map(lambda j: j(), jobs):
            print(line)


def env(gpu):
    import importlib
    vers = []
    for mod in ("torch", "torchvision", "huggingface_hub", "numpy", "PIL", "cv2"):
        try:
            vers.append(f"{mod} {getattr(importlib.import_module(mod), '__version__', '?')}")
        except Exception as ex:
            print(f"!!! [env] {mod} 불러오기 실패 ({type(ex).__name__})")
            return 1
    print("[env] " + " · ".join(vers))
    import torch
    if gpu:
        if not torch.cuda.is_available():
            print("!!! [env] GPU 를 못 봄")
            return 1
        x = torch.randn(1024, 1024, device="cuda")
        y = (x @ x).sum().item()
        free, total = torch.cuda.mem_get_info()
        print(f"[env] {torch.cuda.get_device_name(0)} · bf16 {torch.cuda.is_bf16_supported()} · 메모리 {gib(free)} / {gib(total)} · 연산 확인 {y == y}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["box", "env"])
    ap.add_argument("--app", default="/app")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--links", default="")
    a = ap.parse_args()
    sys.exit(box(a.app, a.links) or 0 if a.what == "box" else env(a.gpu))
