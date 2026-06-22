#!/usr/bin/env python3
"""
fetch_ocr_models.py — Tải sẵn model PaddleOCR để OCR Max chạy OFFLINE.

CHẠY 1 LẦN trên máy CÓ internet, sau đó COPY thư mục models/ sang máy air-gapped
(đặt cạnh app / file .exe).

    python tools/fetch_ocr_models.py                 # lang vi (tiếng Việt)
    python tools/fetch_ocr_models.py --lang en

Model lưu vào: <project>/models/paddle/{det,rec,cls}/
"""
from __future__ import annotations
import argparse
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # <project> (cha của tools/)
PADDLE_DIR = os.path.join(ROOT, "models", "paddle")


def fetch_paddle(lang: str = "vi") -> None:
    try:
        from paddleocr import PaddleOCR
    except ImportError:
        print("Chưa cài PaddleOCR. Chạy: pip install paddlepaddle paddleocr")
        return
    print(f"Khởi tạo PaddleOCR(lang={lang}) để tải model về cache…")
    try:
        try:
            PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)
        except TypeError:
            PaddleOCR(lang=lang)                  # PaddleOCR 3.x đổi tham số
    except Exception as e:
        print(f"Tải lỗi (kiểm tra mạng): {e}")
        return
    # Copy model det/rec/cls từ ~/.paddleocr → models/paddle/
    cache = os.path.join(os.path.expanduser("~"), ".paddleocr")
    found = {"det": None, "rec": None, "cls": None}
    for root, _dirs, files in os.walk(cache):
        if any(f.startswith("inference.pd") for f in files):
            low = root.replace("\\", "/").lower()
            for kind in found:
                if "/" + kind in low and found[kind] is None:
                    found[kind] = root
    os.makedirs(PADDLE_DIR, exist_ok=True)
    for kind, srcd in found.items():
        if not srcd:
            print(f"! chưa thấy model {kind} trong {cache} "
                  "(một số bản không dùng 'cls' — thiếu cls vẫn chạy được)")
            continue
        dstd = os.path.join(PADDLE_DIR, kind)
        shutil.rmtree(dstd, ignore_errors=True)
        shutil.copytree(srcd, dstd)
        print(f"✓ {kind} → {dstd}")
    print(f"\nXONG. Copy thư mục models/ sang máy offline. Model ở: {PADDLE_DIR}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Tải model PaddleOCR cho chế độ offline.")
    ap.add_argument("--lang", default="vi", help="mã ngôn ngữ PaddleOCR (vi/en/japan…)")
    a = ap.parse_args()
    print(f"Project: {ROOT}")
    fetch_paddle(a.lang)


if __name__ == "__main__":
    main()
