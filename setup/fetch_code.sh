#!/usr/bin/env bash
# 공식 코드 받기 (H200·로컬 공용, git 없이): 저장소를 고정 커밋의 GitHub 압축본 (codeload.github.com/<저장소>/tar.gz/<커밋>) 으로 받아,
# 쓰는 파일만 <dest> 에 풀고 파일마다 목록 (sha256) 과 대조한다.
#   dav2 (기본)  DepthAnything/Depth-Anything-V2 — metric_depth/depth_anything_v2/ · LICENSE (setup/code.sha256)
#   unidepth     lpiccinelli-eth/UniDepth — unidepth/ 꾸러미 · LICENSE (setup/code_unidepth.sha256). 선생 점검 (run.sh teacher) 과 팔 ours 의 라벨 뽑기 (src/labels.py) 에 쓴다. 하나라도 다르거나 빠지면 실패. 이미 맞게 받았으면 건너뛴다 (.commit). 끊기면 30 초 쉬고 3 번까지.
# 이 코드는 저장소에 넣지 않고 고치지 않는다 — 우리 코드 (src/model.py · src/teacher_check.py) 가 불러 쓴다.
# 사용: bash setup/fetch_code.sh <dest> [dav2|unidepth]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
DEST=${1:?"받을 폴더를 준다"}
case ${2:-dav2} in
  dav2) REPO_NAME=DepthAnything/Depth-Anything-V2; COMMIT=a561b849ebae10a6f5ef49e26c83cbbcd36c71bf; LIST=$HERE/code.sha256;;
  unidepth) REPO_NAME=lpiccinelli-eth/UniDepth; COMMIT=8d8cfe4c7ee15297099983607febf0d4f32eb3d6; LIST=$HERE/code_unidepth.sha256;;
  *) echo "!!! [code] 모르는 이름: $2 (dav2 | unidepth)" >&2; exit 2;;
esac
export PYTHONDONTWRITEBYTECODE=1

verify() {
  python3 -I -B - "$DEST" "$LIST" <<'PY'
import hashlib, os, sys
root, lst = sys.argv[1:]
want = [l.split(None, 1) for l in open(lst, encoding="utf-8") if l.strip() and not l.startswith("#")]
bad = []
for h, p in want:
    p = p.strip()
    f = os.path.join(root, p)
    if not os.path.isfile(f):
        bad.append("없음 " + p)
        continue
    with open(f, "rb") as fh:
        if hashlib.sha256(fh.read()).hexdigest() != h:
            bad.append("다름 " + p)
if bad:
    print(f"!!! [code] 대조 실패 {len(bad)} / {len(want)}: " + ", ".join(bad[:5]))
    sys.exit(1)
print(f"[code] 대조 통과: 파일 {len(want)} 개")
PY
}

if [ "$(cat "$DEST/.commit" 2>/dev/null)" = "$COMMIT" ] && verify; then exit 0; fi
for a in 1 2 3; do
  rm -rf "${DEST:?}"; mkdir -p "$DEST"
  python3 -I -B - "$DEST" "https://codeload.github.com/$REPO_NAME/tar.gz/$COMMIT" "$LIST" <<'PY' && break
import os, sys, tarfile, urllib.request
dst, url, lst = sys.argv[1:]
want = {l.split(None, 1)[1].strip() for l in open(lst, encoding="utf-8") if l.strip() and not l.startswith("#")}
with urllib.request.urlopen(url, timeout=300) as r, tarfile.open(fileobj=r, mode="r|gz") as t:
    for m in t:                          # 맨 위 폴더 (<저장소>-<커밋>/) 를 떼고, 목록에 있는 정규 파일만 우리가 만든 경로에 쓴다
        parts = m.name.split("/", 1)
        if len(parts) < 2 or not m.isfile() or parts[1] not in want:
            continue
        p = os.path.join(dst, parts[1])
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f:
            f.write(t.extractfile(m).read())
PY
  [ "$a" = 3 ] && { echo "!!! [code] 받기 실패 3 번 ($REPO_NAME @ ${COMMIT:0:8})" >&2; exit 1; }
  echo "[code] 받기 실패 ($a/3) — 30 초 뒤 다시" >&2; sleep 30
done
verify
echo "$COMMIT" > "$DEST/.commit"
echo "[code] $REPO_NAME @ ${COMMIT:0:8} → 파일 $(grep -vc '^#' "$LIST") 개"
