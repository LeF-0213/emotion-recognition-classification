from __future__ import annotations

import argparse, re, shutil
from pathlib import Path

DEFAULT_EMOTIONS = ("happy", "sad", "angry", "neutral")

def _norm(s: str) -> str:
    return s.strip().lower()

def _find_child_case_insensitive(parent: Path, name_hint: str) -> Path | None:
    if not parent.is_dir():
        return None
    want = _norm(name_hint)
    for c in parent.iterdir():
        if c.is_dir() and _norm(c.name) == want:
            return c
    return None

def _read_text_file(path: Path) -> str:
    for enc in ("utf-8", "latin1", "cp1252"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")

def _safe_id(emotion: str, stem: str) -> str:
    base = f"{emotion}__{stem}"
    return re.sub(r"[^\w-]+", "_", base, flags=re.UNICODE)

def find_spectorgram(spec_emotion_dir: Path, stem: str) -> Path | None:
    p = spec_emotion_dir / f"{stem}.jpg"
    if p.is_file():
        return p
    return None

def main():
    ap = argparse.ArgumentParser()
    # 받을 인자 등록(이름, 타입, 기본값, 설명)
    ap.add_argument(
        "--erd-root",
        type=Path,
        required=True,
        help="ERD-MA 폴더 (그 안에 ERD-MA Text, 스펙 폴더 등이 있는 경로)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("dataset"),
        help="출력 루트 (spectrograms/, texts.csv, labels.csv)",
    )
    ap.add_argument(
        "--spec-folder",
        choices=("mel", "spec"),
        default="mel",
        help="mel: ERD-MA Mel-Spectograms, spec: ERD-MA Spectogram",
    )
    ap.add_argument(
        "--text-folder-name",
        default="ERD-MA-Text",
        help="텍스트 루트 폴더 이름",
    )
    ap.add_argument(
        "--mel-folder-name",
        default="ERD-MA-Mel-Spectrograms",
        help="Mel 스펙 폴더 이름",
    )
    ap.add_argument(
        "--spec-folder-name",
        default="ERD-MA-Spectrograms",
        help="일반 스펙 폴더 이름 (오타 포함 원본명)",
    )
    args = ap.parse_args()

    erd = args.erd_root.resolve()
    text_root = erd / args.text_folder_name
    if args.spec_folder == "mel":
        spec_root = erd / args.mel_folder_name
    else:
        spec_root = erd / args.spec_folder_name

    if not text_root.is_dir():
        raise FileNotFoundError(text_root)
    if not spec_root.is_dir():
        raise FileNotFoundError(spec_root)

    out = args.out.resolve()
    spec_out = out / "spectrograms"
    spec_out.mkdir(parents=True, exist_ok=True)

    rows_text = []
    rows_label = []
    missing_spec = []
    missing_pairs = []

    for emotion in DEFAULT_EMOTIONS:
        td = _find_child_case_insensitive(text_root, emotion)
        sd = _find_child_case_insensitive(spec_root, emotion)
        if td is None or sd is None:
            missing_pairs.append(emotion)
            continue

        for tf in sorted(td.iterdir()):
            if not tf.is_file() or tf.suffix.lower() not in '.txt':
                continue
            stem = tf.stem
            sp = find_spectorgram(sd, stem)
            if sp is None:
                missing_spec.append(emotion, stem)
                continue

            sample_id = _safe_id(emotion, stem)
            dst = spec_out / f"{sample_id}.jpg"
            shutil.copy(sp, dst)

            rows_text.append({"id": sample_id, "text": _read_text_file(tf).strip()})
            rows_label.append({"id": sample_id, "emotion": emotion})

    if missing_pairs:
        print("WARN: 감정 폴더 없음", missing_pairs)
    if missing_spec:
        print(f"WARN: 스펙 없음 {len(missing_spec)}건 (예: {missing_spec[:5]})")

    import pandas as pd

    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_text).to_csv(out / "texts.csv", index=False, encoding="utf-8")
    pd.DataFrame(rows_label).to_csv(out / "labels.csv", index=False, encoding="utf-8")
    print(f"완료: texts {len(rows_text)}, labels {len(rows_label)} → {out}")

if __name__ == "__main__":
    main()

