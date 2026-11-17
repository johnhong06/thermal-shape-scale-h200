# 관리자 안내

이 저장소는 열화상 깊이 추정 실험의 H200 실행 묶음입니다. 평소에는 따로 해 주실 일이 없고, 아래 '데이터를 받지 못할 때'만 도움이 필요합니다.

## 실행

요청 이슈의 실행 명령은 한 줄입니다.

```bash
bash run.sh check LINKS=<값>                 # 1–2분, 설치 없음
bash run.sh teacher KIND=smoke LINKS=<값>    # 선생 점검을 작은 데이터로 끝까지 (학습 없음)
bash run.sh teacher LINKS=<값>               # 선생 점검 (학습 없음)
bash run.sh smoke LINKS=<값>                 # 학습 · 평가를 작은 데이터로 끝까지
bash run.sh all LINKS=<값>                   # 본 실험
```

- 저장소, 이슈, 명령 어디에도 토큰이나 비밀번호가 없습니다. 필요한 것은 모두 로그인 없이 받습니다. `LINKS`는 데이터 묶음을 받을 곳의 목록을 가리키는 값이고 자격 증명이 아닙니다.
- 패키지는 작업이 스스로 설치합니다(uv로 만든 가상환경, `/app/scratch/thermal` 아래).
- 쓰는 경로는 `/app/scratch/thermal`(작업 폴더)과 `/app/output/<작업 이름>`(결과)뿐입니다. `/app/data`는 읽기만 합니다.
- 필요한 공간은 작업 폴더 약 60 GB, 결과 1 GB 미만입니다. 명령에 `CKPT=keep`이 있으면 학습한 모델(하나에 1.3 GB, 많으면 9개)을 결과 폴더의 `ckpt/`에 남기므로 약 12 GB가 더 듭니다. 뒤의 평가 작업(`eval` 모드)이 읽는 파일이라 실험이 끝날 때까지 두어 주시기 바랍니다.

## 받는 것

| 대상 | 받는 곳 | 크기 |
|---|---|---|
| 학습 · 평가 데이터 묶음 (선생 점검 모드는 그 가운데 2.8 GiB 만, 추가 쌍 4.0 GiB 는 아직 받지 않음) | Google Drive (링크 공유 파일) | 23.6 GiB |
| Depth Anything V2 ViT-L 가중치 | Hugging Face | 1.3 GB |
| Depth Anything V2 코드 | GitHub (고정 커밋) | 수 MB |
| UniDepthV2 ViT-L 가중치 (`teacher` 모드만) | Hugging Face | 1.4 GB |
| UniDepth 코드 (`teacher` 모드만) | GitHub (고정 커밋) | 22 MB |
| 파이썬 패키지 | PyPI · PyTorch 휠 색인 | 약 6 GB |

## 데이터를 받지 못할 때

`check`나 작업 로그에 `Drive 가 파일 대신 웹 페이지를 줌`이 나오면 Google Drive의 받기 한도에 걸린 것입니다. 이때는 아래 파일을 전달드리겠습니다. `/app/data` 아래 아무 폴더에 **풀지 말고 그대로** 올려 주시면, 작업이 크기와 sha256으로 찾아서 씁니다.

| 파일 | 크기 (바이트) | sha256 |
|---|---|---|
| `smoke.tar` | 45,608,960 | `d197e1fa52e080dbc6280bd15a81c16daa6f9694f4df3adc7265395676c6f168` |
| `train.tar` | 5,741,977,600 | `4f81d06e1b3ca4a6b48683db9e240b3648e98aa237a47819e9f44e92f5432ed9` |
| `val.tar` | 1,110,333,440 | `2769b33f55092cdf69caef739a1564050411dd1e83988df5db11aa9c99b9b84a` |
| `test_day.tar` | 543,303,680 | `13e53a6bd874485539d341348cd418aa0c2175ee48fb6797326f8292de04ac03` |
| `test_night.tar` | 517,969,920 | `894687fbf10ba36323c7a64347c65538e4c8eef7bdf132d858334c0070aa23d4` |
| `test_rainy.tar` | 518,563,840 | `615cb2c7624ffa966449fb0baa294af373afbdf8a65e119e3ee5585ed7d41c72` |
| `firestereo_test.tar` | 2,717,808,640 | `69c571add7aaf2934263b67ea29b585c92b59b745e0bd71e512802dcf31b5855` |
| `vivid.tar` | 1,300,643,840 | `a6097ad4f009ee8345e9655d290fb41971ecaefe10d9b51900d30fff2e1a76d7` |
| `pairs_kaist.tar` | 3,877,734,400 | `dd950c773e077ae235dceb1fbac2535e1c149f1c33335299579f6bfc96cf2bbe` |
| `pairs_llvip.tar` | 766,423,040 | `b95946d2732aa5d2dcecfb1b68e358235b598c368d82222bb7e640e7fa837062` |
| `vivid_day_rgb.tar` | 929,638,400 | `4397f6618559b6dc44ae4acd23cd11c0eace2584c68d983c09b1dddd3ee4ae4c` |
| `ms2_rgb_train.tar` | 1,890,672,640 | `a9ef05b4dbc761baacc3151396ffb42d5c58d5f289451be18931b57c6611c7e1` |
| `ms2_rgb_val.tar` | 1,133,025,280 | `9eef9bf557de5624df52f0f554a92e36d34f9d52e9dcae189dc7aef22589ecd6` |
| `smoke_teacher.tar` | 12,748,800 | `1af59b13ac853181dd6a607d408c59928c10eb4071ea9076a073ff55b3e75820` |
| `pairs_freiburg.tar` | 2,005,514,240 | `c5093158f009191eca5dc9fb036f691b47e0f4b7c290e17c4e19368fbd1a1a9f` |
| `pairs_tartan.tar` | 2,255,431,680 | `27c3137aac69141e94080eec7fa4346afbb97227c34f802382d26c76067c276f` |

올린 뒤 확인:

```bash
sha256sum *.tar
```

## 결과

작업이 끝나면 `/app/output/<작업 이름>/results.zip` 하나를 요청드립니다(`ckpt/` 폴더는 보내지 않으셔도 됩니다). 작업 이름은 화면 마지막 줄(`[run] … 끝 — … 결과 /app/output/…`)에 있습니다. 크기는 수십 MB입니다.

## 끝난 뒤

데이터에 재배포 제한이 있습니다. `/app/data`에 올리신 묶음이 있으면 실험이 끝났다고 말씀드릴 때 지워 주시기 바랍니다.
