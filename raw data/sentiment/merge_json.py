"""Gộp các file JSON đã crawl vào output/final.json, bỏ bài trùng.

    python sentiment/merge_json.py output/posts_all.json
    python sentiment/merge_json.py "output/*/posts.jsonl" output/posts_2026*.json

Quy tắc:
  - Bài trùng LINK (cùng /posts/<id>) hoặc trùng NỘI DUNG (bài đăng lại) với bài đã có -> không thêm,
    chỉ bổ sung các comment mới của nó vào bài cũ (comment trùng = cùng người viết + cùng nội dung).
  - Bài mới -> thêm vào, bỏ comment bị cào lặp trong bài.
  - Comment rỗng (chỉ sticker/ảnh) không được bổ sung.
  - Trước khi ghi, sao lưu final.json cũ thành final_backup_<thời gian>.json.
"""
import argparse
import json
import re
import shutil
import unicodedata
from datetime import datetime
from pathlib import Path

from json_to_csv import load_posts

PROJECT = Path(__file__).parent.parent


def norm(s):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s or "")).strip()


def post_id(p):
    m = re.search(r"/posts/(\d+)", p.get("url") or "")
    return m[1] if m else p.get("url")


def comment_key(c):
    return c.get("author"), norm(c.get("text"))


def merge(final, new_posts):
    by_id = {post_id(p): p for p in final}
    by_text = {norm(p.get("text")): p for p in final if norm(p.get("text"))}
    stats = {"bài_mới": 0, "bài_trùng": 0, "comment_bổ_sung": 0}
    for p in new_posts:
        text = norm(p.get("text"))
        target = by_id.get(post_id(p)) or (by_text.get(text) if text else None)
        if target is None:
            seen, comments = set(), []
            for c in p.get("comments") or []:
                if comment_key(c) not in seen:
                    seen.add(comment_key(c))
                    comments.append(c)
            p = {**p, "comments": comments}
            final.append(p)
            by_id[post_id(p)] = p
            if text:
                by_text[text] = p
            stats["bài_mới"] += 1
            continue
        stats["bài_trùng"] += 1
        have = {comment_key(c) for c in target.get("comments") or []}
        for c in p.get("comments") or []:
            if norm(c.get("text")) and comment_key(c) not in have:
                have.add(comment_key(c))
                target.setdefault("comments", []).append(c)
                stats["comment_bổ_sung"] += 1
    return final, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="file/glob .json hoặc .jsonl cần gộp vào")
    ap.add_argument("--final", default=str(PROJECT / "output" / "final.json"))
    args = ap.parse_args()

    final_path = Path(args.final)
    final = json.loads(final_path.read_text(encoding="utf-8")) if final_path.exists() else []
    n_before = len(final)
    final, stats = merge(final, load_posts(args.inputs))

    if final_path.exists():
        backup = final_path.with_name(f"{final_path.stem}_backup_{datetime.now():%Y%m%d_%H%M%S}.json")
        shutil.copy(final_path, backup)
        print(f"Đã sao lưu {backup.name}")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    final_path.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Bài mới thêm: {stats['bài_mới']} | bài đã có (bỏ qua): {stats['bài_trùng']} | "
          f"comment mới bổ sung vào bài đã có: {stats['comment_bổ_sung']}")
    print(f"{final_path.name}: {n_before} -> {len(final)} bài, "
          f"{sum(len(p.get('comments') or []) for p in final)} comment")


if __name__ == "__main__":
    main()
