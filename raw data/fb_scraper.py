"""
Thu thập bài đăng + bình luận từ một trang/nhóm Facebook bằng Playwright.

Cách dùng:
    pip install playwright
    playwright install chromium

    # Lần đầu: mở trình duyệt để bạn tự đăng nhập Facebook (session được lưu lại)
    python fb_scraper.py --login

    # Cào dữ liệu
    python fb_scraper.py "https://www.facebook.com/groups/xxxx" --max-posts 20
    python fb_scraper.py "https://www.facebook.com/xxxx/posts/yyyy"      # 1 bài cụ thể

    # Lấy TOÀN BỘ bài (lưu dần, Ctrl+C rồi chạy lại lệnh cũ sẽ làm tiếp)
    python fb_scraper.py "https://www.facebook.com/groups/xxxx" --all
    python fb_scraper.py "https://www.facebook.com/groups/xxxx" --all --recollect   # cập nhật bài mới

Kết quả lưu ở output/posts_<timestamp>.json và .csv
(chế độ --all: output/<tên nhóm>/posts_all.json và comments_all.csv)
"""

import argparse
import csv
import json
import random
import re
import time
from datetime import datetime
from pathlib import Path

from playwright.sync_api import sync_playwright, Page, Locator, TimeoutError as PWTimeout

PROFILE_DIR = Path(__file__).parent / "fb_profile"  # lưu cookie đăng nhập
OUTPUT_DIR = Path(__file__).parent / "output"

# Text của các nút mở rộng (hỗ trợ giao diện tiếng Việt + tiếng Anh)
SEE_MORE = ["Xem thêm", "See more"]
MORE_COMMENTS = [
    "Xem thêm bình luận", "View more comments",
    "Xem các bình luận trước", "View previous comments",
    "câu trả lời", "phản hồi", "repl",  # "Xem 3 câu trả lời", "View 3 replies"
]
ALL_COMMENTS_FILTER = ["Tất cả bình luận", "All comments"]
POST_ID_RE = re.compile(r"(https://www\.facebook\.com/(?:groups/[^/]+|[^/?]+)/(?:posts|permalink)/[\w.]+)")
POST_LINK_RE = re.compile(r"/(posts|permalink|videos|photos|reel)/|story_fbid=|/permalink\.php|multi_permalinks=")


def human_pause(a=0.8, b=2.0):
    time.sleep(random.uniform(a, b))


def launch(p, headless=False):
    opts = dict(
        user_data_dir=str(PROFILE_DIR),
        headless=headless,
        viewport={"width": 1280, "height": 900},
        locale="vi-VN",
        # Ẩn dấu hiệu tự động hoá, nếu không FB hay treo ở trang đăng nhập / 2FA
        args=["--disable-blink-features=AutomationControlled"],
        ignore_default_args=["--enable-automation"],
    )
    try:
        ctx = p.chromium.launch_persistent_context(channel="chrome", **opts)  # Chrome thật đã cài
    except Exception:
        ctx = p.chromium.launch_persistent_context(**opts)  # fallback: Chromium của Playwright
    ctx.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    return ctx


def import_cookies(path: str):
    """Nạp cookie xuất từ trình duyệt thường (extension Cookie-Editor -> Export JSON)."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    same_site = {"no_restriction": "None", "unspecified": "Lax", "lax": "Lax", "strict": "Strict"}
    cookies = []
    for c in raw:
        ck = {
            "name": c["name"], "value": c["value"],
            "domain": c.get("domain", ".facebook.com"), "path": c.get("path", "/"),
            "secure": c.get("secure", True), "httpOnly": c.get("httpOnly", False),
            "sameSite": same_site.get(str(c.get("sameSite", "")).lower(), "None"),
        }
        exp = c.get("expirationDate") or c.get("expires")
        if exp:
            ck["expires"] = float(exp)
        cookies.append(ck)
    with sync_playwright() as p:
        ctx = launch(p)
        ctx.add_cookies(cookies)
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.goto("https://www.facebook.com/")
        input(">> Kiểm tra đã vào được Facebook chưa, rồi nhấn Enter...")
        ctx.close()
    print("Đã nạp", len(cookies), "cookie vào", PROFILE_DIR)


def do_login():
    with sync_playwright() as p:
        ctx = launch(p)
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.goto("https://www.facebook.com/")
        input(">> Đăng nhập Facebook trong cửa sổ trình duyệt, xong thì nhấn Enter ở đây...")
        ctx.close()
    print("Đã lưu session vào", PROFILE_DIR)


def click_all(scope: Locator | Page, texts: list[str], max_rounds=30):
    """Bấm lặp lại mọi nút có chứa các đoạn text cho tới khi không còn."""
    for _ in range(max_rounds):
        clicked = False
        for t in texts:
            btns = scope.locator('div[role="button"], span[role="button"]').filter(has_text=t)
            for i in range(btns.count()):
                b = btns.nth(i)
                try:
                    if b.is_visible():
                        b.click(timeout=2000)
                        clicked = True
                        human_pause(0.5, 1.2)
                except Exception:
                    pass
        if not clicked:
            return


def switch_to_all_comments(page: Page):
    """Đổi bộ lọc 'Phù hợp nhất' -> 'Tất cả bình luận' nếu có."""
    try:
        f = page.locator('div[role="button"]').filter(
            has_text=re.compile(r"Phù hợp nhất|Most relevant|Mới nhất|Newest"))
        if f.count():
            f.first.click(timeout=3000)
            human_pause()
            opt = page.locator('div[role="menuitem"]').filter(
                has_text=re.compile("|".join(ALL_COMMENTS_FILTER)))
            if opt.count():
                opt.first.click(timeout=3000)
                human_pause(1.5, 2.5)
    except Exception:
        pass


def collect_post_links(page: Page, url: str, max_posts: int, max_scrolls: int,
                       links_file: Path | None = None) -> list[str]:
    # Nhóm: sắp xếp theo bài mới nhất để cuộn được hết thay vì "phù hợp nhất"
    if "/groups/" in url and "sorting_setting" not in url:
        url += ("&" if "?" in url else "?") + "sorting_setting=CHRONOLOGICAL"
    page.goto(url, wait_until="domcontentloaded")
    human_pause(3, 4)
    links: list[str] = []
    stale = 0
    for n in range(max_scrolls):
        reveal_hidden_links(page)
        hrefs = page.eval_on_selector_all(
            'div[role="feed"] a[href], div[role="main"] a[href]',
            "els => els.map(e => e.href)",
        )
        new = []
        for h in hrefs:
            if not POST_LINK_RE.search(h):
                continue
            m = POST_ID_RE.search(h)  # bỏ ?comment_id=..., ?__cft__... -> link gốc của bài
            clean = m.group(1) + "/" if m else h.split("&__cft__")[0]
            if clean not in links and clean not in new:
                new.append(clean)
        links += new
        if links_file and new:
            with links_file.open("a", encoding="utf-8") as f:
                f.writelines(l + "\n" for l in new)
        if n % 10 == 0 or new:
            print(f"  cuộn {n}: đã có {len(links)} link bài", end="\r")
        if len(links) >= max_posts:
            break
        # Hết bài thật: nhiều lần cuộn liên tiếp không ra link mới
        stale = stale + 1 if not new else 0
        if stale >= 12:
            break
        page.mouse.wheel(0, 2500)
        human_pause(1.5, 3) if stale < 3 else human_pause(4, 7)  # chờ lâu hơn khi FB load chậm
    print()

    if not links:
        OUTPUT_DIR.mkdir(exist_ok=True)
        page.screenshot(path=str(OUTPUT_DIR / "debug.png"), full_page=False)
        (OUTPUT_DIR / "debug.html").write_text(page.content(), encoding="utf-8")
        print(f"!! Không thấy bài nào. URL hiện tại: {page.url} | title: {page.title()}")
        print("   Đã lưu output/debug.png và output/debug.html để kiểm tra")
    return links[:max_posts]


def reveal_hidden_links(page: Page):
    """FB để href='#' cho link thời gian đăng bài, chỉ gán link thật khi hover -> hover hết."""
    # Chỉ hover link chưa xử lý (đánh dấu data-fbh) để không chậm dần khi feed dài ra
    # Bỏ qua link nằm trong bình luận (chúng cũng có dạng ?__cft__ nhưng không cần)
    skip = ':not([data-fbh]):not([aria-label^="Bình luận"] a):not([aria-label^="Comment"] a)' \
           ':not([aria-label^="Phản hồi"] a):not([aria-label^="Reply"] a)'
    anchors = page.locator(
        f'div[role="feed"] a[role="link"][href^="?"]{skip}, '
        f'div[role="feed"] a[role="link"][href^="#"]{skip}')
    for _ in range(min(anchors.count(), 60)):
        a = anchors.first
        try:
            a.evaluate("e => e.setAttribute('data-fbh', '1')")
            if a.is_visible():
                a.hover(timeout=1000)
                time.sleep(0.15)
        except Exception:
            pass


def extract_comments(page: Page) -> list[dict]:
    """Bình luận trên FB là các [role=article] có aria-label 'Bình luận của ...' / 'Comment by ...'."""
    return page.evaluate(r"""
    () => {
      const out = [];
      const arts = document.querySelectorAll('div[role="article"][aria-label]');
      for (const a of arts) {
        const label = a.getAttribute('aria-label') || '';
        if (!/^(Bình luận|Phản hồi|Comment|Reply)/i.test(label)) continue;
        const isReply = /^(Phản hồi|Reply)/i.test(label);
        const authorEl = a.querySelector('a[role="link"] span, a span');
        const textEl = a.querySelector('div[dir="auto"]');
        const textParts = [...a.querySelectorAll('div[dir="auto"]')].map(d => d.innerText.trim()).filter(Boolean);
        const timeEl = [...a.querySelectorAll('a[href*="comment_id"]')].pop();
        out.push({
          author: authorEl ? authorEl.innerText.trim() : null,
          text: [...new Set(textParts)].join('\n'),
          is_reply: isReply,
          time: timeEl ? timeEl.innerText.trim() : null,
          link: timeEl ? timeEl.href.split('&__cft__')[0] : null,
          aria_label: label,
        });
      }
      return out;
    }
    """)


def scrape_post(page: Page, url: str) -> dict:
    page.goto(url, wait_until="domcontentloaded")
    human_pause(3, 4)

    # Nếu bài mở dạng dialog thì dùng dialog làm scope, ngược lại dùng main
    dialog = page.locator('div[role="dialog"]')
    scope = dialog.last if dialog.count() else page.locator('div[role="main"]')

    click_all(scope, SEE_MORE, max_rounds=3)
    switch_to_all_comments(page)
    click_all(scope, MORE_COMMENTS)
    click_all(scope, SEE_MORE, max_rounds=5)  # mở rộng comment dài

    post = page.evaluate(r"""
    (root) => {
      root = root || document;
      // Nội dung bài: khối data-ad-preview="message" hoặc data-ad-comet-preview
      const msg = root.querySelector('[data-ad-preview="message"], [data-ad-comet-preview="message"]');
      // Tác giả: đi ngược lên từ khối nội dung, lấy link người đăng gần nhất
      // (tránh lấy nhầm tên nhóm ở tiêu đề trang)
      let author = null;
      for (let el = msg; el && el !== root.parentElement && !author; el = el.parentElement) {
        author = [...el.querySelectorAll('a[href*="/user/"], h2 a, h3 a, strong a')]
          .find(a => a.innerText.trim() && !/\/groups\/[^/]+\/?(\?|$)/.test(a.getAttribute('href') || ''));
      }
      author = author || root.querySelector('h2 a, h3 a, strong a');
      const imgs = [...root.querySelectorAll('img')]
        .map(i => i.src).filter(s => s.includes('scontent'));
      return {
        author: author ? author.innerText.trim() : null,
        text: msg ? msg.innerText.trim() : null,
        images: [...new Set(imgs)].slice(0, 20),
      };
    }
    """, scope.element_handle() if scope.count() else None)

    # Bài mở dạng dialog có tiêu đề "Bài viết của X" / "X's post" -> tác giả chính xác nhất
    try:
        heading = scope.locator("h2, h3").first.inner_text(timeout=1500).strip()
        m = re.match(r"Bài viết của (.+)|(.+?)['’]s post", heading)
        if m:
            post["author"] = (m.group(1) or m.group(2)).strip()
    except Exception:
        pass

    # Fallback nội dung: lấy toàn bộ text khối đầu tiên nếu không thấy message
    if not post.get("text"):
        try:
            post["text"] = scope.locator('div[dir="auto"]').first.inner_text(timeout=2000)
        except PWTimeout:
            pass

    post["url"] = url
    post["comments"] = extract_comments(page)
    post["scraped_at"] = datetime.now().isoformat(timespec="seconds")
    return post


def save(posts: list[dict], out_dir: Path = OUTPUT_DIR, name: str | None = None):
    out_dir.mkdir(parents=True, exist_ok=True)
    name = name or datetime.now().strftime("%Y%m%d_%H%M%S")
    jpath = out_dir / f"posts_{name}.json"
    jpath.write_text(json.dumps(posts, ensure_ascii=False, indent=2), encoding="utf-8")

    cpath = out_dir / f"comments_{name}.csv"
    with cpath.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["post_url", "post_author", "post_text", "comment_author", "comment_text", "is_reply", "comment_time"])
        for p in posts:
            if not p["comments"]:
                w.writerow([p["url"], p["author"], p["text"], "", "", "", ""])
            for c in p["comments"]:
                w.writerow([p["url"], p["author"], p["text"], c["author"], c["text"], c["is_reply"], c["time"]])
    print(f"Đã lưu: {jpath}\n        {cpath}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url", nargs="?", help="Link trang / nhóm / bài viết Facebook")
    ap.add_argument("--login", action="store_true", help="Mở trình duyệt để đăng nhập thủ công")
    ap.add_argument("--cookies", help="File JSON cookie xuất từ trình duyệt thường (Cookie-Editor)")
    ap.add_argument("--max-posts", type=int, default=None,
                    help="Số bài tối đa (mặc định 10; với --all mặc định là không giới hạn)")
    ap.add_argument("--max-scrolls", type=int, default=None,
                    help="Số lần cuộn tối đa (mặc định tự tính = 2 x max-posts, tối thiểu 40)")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--all", action="store_true",
                    help="Lấy toàn bộ bài: lưu dần vào output/<tên nhóm>/, chạy lại sẽ tiếp tục chỗ cũ")
    ap.add_argument("--recollect", action="store_true",
                    help="(dùng với --all) cuộn lại feed để tìm thêm bài mới dù đã có links.txt")
    ap.add_argument("--rest-every", type=int, default=30, help="Nghỉ dài sau mỗi N bài")
    args = ap.parse_args()

    if args.login:
        return do_login()
    if args.cookies:
        return import_cookies(args.cookies)
    if not args.url:
        ap.error("cần truyền url, --login hoặc --cookies")
    if args.all:
        return scrape_all(args)
    args.max_posts = args.max_posts or 10
    args.max_scrolls = args.max_scrolls or max(40, args.max_posts * 2)

    with sync_playwright() as p:
        ctx = launch(p, headless=args.headless)
        page = ctx.pages[0] if ctx.pages else ctx.new_page()

        if POST_LINK_RE.search(args.url):
            links = [args.url]
        else:
            links = collect_post_links(page, args.url, args.max_posts, args.max_scrolls)
        print(f"Tìm thấy {len(links)} bài đăng")

        posts = []
        for i, link in enumerate(links, 1):
            print(f"[{i}/{len(links)}] {link}")
            try:
                post = scrape_post(page, link)
                print(f"    -> {len(post['comments'])} bình luận")
                posts.append(post)
            except Exception as e:
                print("    lỗi:", e)
            human_pause(2, 5)  # giãn cách để tránh bị FB chặn

        save(posts)
        ctx.close()


def scrape_all(args):
    """Chế độ lấy toàn bộ, có checkpoint:
       output/<slug>/links.txt   - link bài tìm được (ghi dần khi cuộn)
       output/<slug>/posts.jsonl - mỗi dòng 1 bài đã cào (ghi ngay sau mỗi bài)
    Bị dừng (Ctrl+C, mất mạng, FB chặn...) thì chạy lại đúng lệnh cũ để làm tiếp."""
    slug = re.sub(r"[^\w.-]+", "_", args.url.rstrip("/").split("facebook.com/")[-1].split("?")[0])
    job = OUTPUT_DIR / slug
    job.mkdir(parents=True, exist_ok=True)
    links_file, posts_file = job / "links.txt", job / "posts.jsonl"

    def read_lines(f: Path) -> list[str]:
        return [l.strip() for l in f.read_text(encoding="utf-8").splitlines() if l.strip()] if f.exists() else []

    done = {json.loads(l)["url"] for l in read_lines(posts_file)}

    with sync_playwright() as p:
        ctx = launch(p, headless=args.headless)
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        try:
            # Bước 1: cuộn feed tới cuối để gom link (bỏ qua nếu đã có từ lần chạy trước)
            collected_flag = job / "links.done"
            if not collected_flag.exists() or args.recollect:
                print("Bước 1: cuộn feed để gom link bài (có thể mất khá lâu)...")
                collect_post_links(page, args.url, args.max_posts or 10**9,
                                   args.max_scrolls or 10**6, links_file)
                collected_flag.touch()
            links = list(dict.fromkeys(read_lines(links_file)))  # bỏ trùng giữa các lần chạy
            todo = [l for l in links if l not in done]
            print(f"Tổng {len(links)} bài | đã cào {len(done)} | còn {len(todo)}")

            # Bước 2: cào từng bài, ghi ngay vào posts.jsonl
            for i, link in enumerate(todo, 1):
                print(f"[{i}/{len(todo)}] {link}")
                for attempt in (1, 2):
                    try:
                        post = scrape_post(page, link)
                        with posts_file.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(post, ensure_ascii=False) + "\n")
                        print(f"    -> {len(post['comments'])} bình luận")
                        break
                    except Exception as e:
                        print(f"    lỗi (lần {attempt}):", str(e).splitlines()[0])
                        # Tab bị crash (thường do video/ảnh nặng) -> mở tab mới để không hỏng các bài sau
                        try:
                            page.close()
                        except Exception:
                            pass
                        page = ctx.new_page()
                        human_pause(3, 5)
                else:
                    with (job / "failed.txt").open("a", encoding="utf-8") as f:
                        f.write(link + "\n")
                if "checkpoint" in page.url or "login" in page.url:
                    print("!! FB yêu cầu xác minh / đăng nhập lại -> dừng. Xử lý xong chạy lại lệnh để tiếp tục.")
                    break
                if i % args.rest_every == 0:
                    rest = random.uniform(60, 180)
                    print(f"   nghỉ {rest:.0f}s cho đỡ bị FB để ý...")
                    time.sleep(rest)
                else:
                    human_pause(3, 7)
        except KeyboardInterrupt:
            print("\nĐã dừng (Ctrl+C). Chạy lại lệnh để tiếp tục.")
        finally:
            try:
                ctx.close()
            except Exception:
                pass

    # Xuất file JSON + CSV gộp từ toàn bộ dữ liệu đã cào
    posts = [json.loads(l) for l in read_lines(posts_file)]
    if posts:
        save(posts, job, "all")


if __name__ == "__main__":
    main()
