"""Chuyển file JSON đã crawl sang CSV chưa có nhãn: mỗi post và mỗi comment là 1 dòng (1 mẫu).

    python sentiment/json_to_csv.py                                   # mặc định: output/final.json
    python sentiment/json_to_csv.py output/final.json "output/*/posts.jsonl" --out data/samples.csv

Cột: id, source (post/comment), text, author, is_reply, comment_time, post_id, url
  - post_id: id của post chứa mẫu -> nhóm được comment theo post
  - Bỏ mẫu có text rỗng (comment chỉ có sticker/ảnh)
  - Bỏ comment bị cào lặp trong cùng post (cùng người viết + cùng nội dung)
"""
import argparse
import glob
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
PROJECT = ROOT.parent

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_SPACE_RE = re.compile(r"\s+")


def clean_text(text):
    text = unicodedata.normalize("NFC", text or "")
    text = _URL_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


def load_posts(patterns):
    posts = []
    for pat in patterns:
        files = sorted(glob.glob(pat))
        if not files:
            print(f"Không tìm thấy file: {pat}")
        for f in files:
            with open(f, encoding="utf-8") as fh:
                posts += [json.loads(l) for l in fh if l.strip()] if f.endswith(".jsonl") else json.load(fh)
    return posts


def post_id(url):
    m = re.search(r"/posts/(\d+)", url or "")
    return m[1] if m else url


def to_rows(posts):
    rows, seen_posts = [], set()
    stats = {"post_trùng": 0, "post_rỗng": 0, "comment_rỗng": 0, "comment_lặp": 0}
    for p in posts:
        pid, url = post_id(p.get("url")), p.get("url", "")
        if pid in seen_posts:  # cùng post xuất hiện ở nhiều file
            stats["post_trùng"] += 1
            continue
        seen_posts.add(pid)

        text = clean_text(p.get("text"))
        if text:
            rows.append({"source": "post", "text": text, "author": p.get("author", ""),
                         "is_reply": None, "comment_time": None, "post_id": pid, "url": url})
        else:
            stats["post_rỗng"] += 1

        seen_comments = set()
        for c in p.get("comments") or []:
            ctext = clean_text(c.get("text"))
            if not ctext:
                stats["comment_rỗng"] += 1
                continue
            key = (c.get("author"), ctext)
            if key in seen_comments:
                stats["comment_lặp"] += 1
                continue
            seen_comments.add(key)
            rows.append({"source": "comment", "text": ctext, "author": c.get("author", ""),
                         "is_reply": c.get("is_reply"), "comment_time": c.get("time"),
                         "post_id": pid, "url": c.get("link") or url})
    df = pd.DataFrame(rows)
    df.insert(0, "id", range(1, len(df) + 1))
    return df, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="*", default=[str(PROJECT / "output" / "final.json")],
                    help="file/glob .json hoặc .jsonl (mặc định output/final.json)")
    ap.add_argument("--out", default=str(ROOT / "data" / "samples.csv"))
    args = ap.parse_args()

    posts = load_posts(args.inputs)
    df, stats = to_rows(posts)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")  # utf-8-sig để Excel đọc đúng tiếng Việt

    counts = df["source"].value_counts()
    print(f"Đọc {len(posts)} post từ {len(args.inputs)} nguồn")
    print(f"-> {len(df)} mẫu: {counts.get('post', 0)} post + {counts.get('comment', 0)} comment")
    print("Đã bỏ:", stats)
    print(f"Đã ghi {args.out}")


if __name__ == "__main__":
    main()
