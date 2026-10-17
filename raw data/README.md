# Phân tích cảm xúc bài đăng UIT Confessions

Dự án thu thập bài đăng và bình luận từ nhóm Facebook [UIT Confessions](https://www.facebook.com/groups/uiter.confessions), rồi phân loại cảm xúc của từng bài và từng bình luận thành **negative / neutral / positive** bằng model [ViSoBERT](https://huggingface.co/uitnlp/visobert).

Model được train qua 2 giai đoạn:

1. **Full fine-tune** ViSoBERT trên 2 bộ dữ liệu công khai: **UIT-VSMEC** (bình luận Facebook) và **UIT-VSFC** (phản hồi của sinh viên UIT).
2. **Fine-tune LoRA** trên 562 câu confession UIT được gán nhãn tay.

### Kết quả chính

| Model | Tập đánh giá | Accuracy | Macro-F1 |
|---|---|---|---|
| Model gốc (VSMEC + VSFC) | Test VSMEC (693 câu) | 0,762 | 0,733 |
| Model gốc (VSMEC + VSFC) | Test VSFC (3.166 câu) | 0,926 | 0,821 |
| Model gốc, **chưa** fine-tune trên dữ liệu UIT | Test UIT (113 câu) | 0,673 | 0,665 |
| **Model cuối (+ LoRA trên dữ liệu UIT)** | **Test UIT (113 câu)** | **0,938** | **0,938** |

Fine-tune LoRA trên vài trăm câu UIT tăng macro-F1 trên dữ liệu confession từ **0,665 lên 0,938** (+27 điểm), mà chỉ train khoảng 2% tham số. Chi tiết và các hạn chế của con số này ở [mục 12](#12-báo-cáo-kết-quả).

---

## Mục lục

1. [Quy trình tổng quát](#1-quy-trình-tổng-quát)
2. [Cấu trúc thư mục](#2-cấu-trúc-thư-mục)
3. [Cài đặt](#3-cài-đặt)
4. [Bước 1: Crawl dữ liệu Facebook](#4-bước-1-crawl-dữ-liệu-facebook)
5. [Bước 2: Gộp dữ liệu vào final.json](#5-bước-2-gộp-dữ-liệu-vào-finaljson)
6. [Bước 3: Chuyển JSON sang CSV](#6-bước-3-chuyển-json-sang-csv)
7. [Bước 4: Train model gốc trên VSMEC + VSFC](#7-bước-4-train-model-gốc-trên-vsmec--vsfc)
8. [Bước 5: Gán nhãn dữ liệu UIT](#8-bước-5-gán-nhãn-dữ-liệu-uit)
9. [Bước 6: Fine-tune LoRA](#9-bước-6-fine-tune-lora)
10. [Bước 7: Dự đoán](#10-bước-7-dự-đoán)
11. [Công cụ phụ: kiểm tra chất lượng nhãn](#11-công-cụ-phụ-kiểm-tra-chất-lượng-nhãn)
12. [Báo cáo kết quả](#12-báo-cáo-kết-quả)
13. [Lỗi thường gặp](#13-lỗi-thường-gặp)
14. [Hạn chế và hướng phát triển](#14-hạn-chế-và-hướng-phát-triển)
15. [Lưu ý an toàn và quyền riêng tư](#15-lưu-ý-an-toàn-và-quyền-riêng-tư)

---

## 1. Quy trình tổng quát

```
 Facebook ──fb_scraper.py──► output/posts_*.json, output/<nhóm>/posts.jsonl
                                   │
                         merge_json.py (gộp, bỏ trùng)
                                   ▼
                          output/final.json
                                   │
                          json_to_csv.py
                                   ▼
                sentiment/data/samples.csv  (mỗi post / comment = 1 mẫu, chưa nhãn)
                                   │
              gán nhãn tay (Bước 5) │
                                   ▼
                sentiment/data/finetune.csv  (562 câu có nhãn)
                                   │
 UIT-VSMEC + UIT-VSFC ──train_vsmec.py──► models/visobert_vsmec_vsfc  (model gốc)
                                   │
                         finetune_lora.py
                                   ▼
                          models/visobert_uit  (model cuối)
                                   │
                            predict.py
                                   ▼
          *_predicted.csv  +  *_predicted_posts.csv (tổng hợp theo post)
```

| Bước | Lệnh | Thời gian (RTX 3060 Laptop) |
|---|---|---|
| 1. Crawl | `python fb_scraper.py <url> --all` | ~14 bài/phút |
| 2. Gộp | `python sentiment/merge_json.py <file mới>` | vài giây |
| 3. JSON sang CSV | `python sentiment/json_to_csv.py` | vài giây |
| 4. Train model gốc | `python sentiment/train_vsmec.py` | ~5–8 phút |
| 5. Gán nhãn | làm tay | — |
| 6. Fine-tune LoRA | `python sentiment/finetune_lora.py` | ~1 phút |
| 7. Dự đoán | `python sentiment/predict.py` | vài giây / 1.000 câu |

---

## 2. Cấu trúc thư mục

```
.
├── README.md
├── fb_scraper.py               # Bước 1: crawl Facebook bằng Playwright
├── fb_profile/                 # phiên đăng nhập Chrome (KHÔNG đẩy lên git)
├── cookies.json                # cookie tạm khi đăng nhập (KHÔNG đẩy lên git, xoá sau khi dùng)
├── output/                     # dữ liệu crawl (KHÔNG đẩy lên git)
│   ├── final.json              # toàn bộ bài + comment đã gộp, bỏ trùng
│   └── final_backup_*.json     # bản sao lưu tự tạo mỗi lần gộp
└── sentiment/
    ├── merge_json.py           # Bước 2
    ├── json_to_csv.py          # Bước 3
    ├── train_vsmec.py          # Bước 4
    ├── finetune_lora.py        # Bước 6
    ├── predict.py              # Bước 7
    ├── find_label_issues.py    # kiểm tra nhãn bằng cleanlab
    ├── visobert_tokenizer.py   # tokenizer ViSoBERT (xem mục 13)
    ├── common.py               # hàm dùng chung cho find_label_issues.py
    ├── data/                   # (KHÔNG đẩy lên git)
    │   ├── samples.csv             # mẫu chưa nhãn
    │   ├── finetune.csv            # mẫu đã gán nhãn tay
    │   ├── samples_predicted.csv   # kết quả dự đoán
    │   └── samples_predicted_posts.csv
    └── models/                 # (KHÔNG đẩy lên git, mỗi model ~390MB)
        ├── visobert_vsmec_vsfc/    # model gốc (Bước 4)
        ├── visobert_uit_lora/      # adapter LoRA (Bước 6)
        └── visobert_uit/           # model cuối = gốc + LoRA đã gộp (Bước 6)
```

Mọi lệnh trong README chạy từ thư mục gốc của repo. Nếu đang đứng trong `sentiment/` thì bỏ tiền tố `sentiment/` và đổi `output/...` thành `../output/...`.

---

## 3. Cài đặt

**Yêu cầu:**
- Python 3.10 trở lên (đã chạy trên 3.12).
- Google Chrome. Nếu không có, scraper dùng Chromium của Playwright.
- GPU NVIDIA (CUDA) cho Bước 4 và 6. Không có GPU vẫn chạy được nhưng rất chậm; khi đó có thể train trên Google Colab hoặc Kaggle.

```bash
pip install playwright pandas scikit-learn
playwright install chromium

pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install "transformers>=5" "huggingface-hub>=1.5,<2" datasets peft sentencepiece

# Chỉ cần cho find_label_issues.py
pip install cleanlab pyvi
```

Phiên bản đã kiểm tra: `torch 2.13 (cu126)`, `transformers 5.17`, `huggingface-hub 1.33`, `datasets 4.4.1`, `peft 0.19`, `sentencepiece 0.2.0`, `scikit-learn 1.7.2`, `pandas 2.2.2`, `cleanlab 2.9`, `playwright 1.63`.

> ⚠️ `transformers 5.17` yêu cầu `huggingface-hub` **dưới bản 2.0**. Một số gói khác (ví dụ cleanlab) có thể tự nâng `huggingface-hub` lên 2.x khi cài, làm `transformers` báo lỗi. Gặp lỗi này thì chạy lại `pip install "huggingface-hub>=1.5,<2"`.

Trên Windows, nếu terminal in tiếng Việt bị lỗi (`UnicodeEncodeError ... cp1258`), đặt biến môi trường trước khi chạy:

```bat
set PYTHONIOENCODING=utf-8              :: cmd
$env:PYTHONIOENCODING="utf-8"           # PowerShell
```

---

## 4. Bước 1: Crawl dữ liệu Facebook

`fb_scraper.py` dùng Playwright để lấy bài đăng và toàn bộ bình luận từ nhóm, trang hoặc một bài viết mà tài khoản của bạn xem được.

### 4.1. Đăng nhập (làm 1 lần)

Script không lưu mật khẩu. Phiên đăng nhập được giữ trong `fb_profile/` và dùng lại cho các lần sau.

**Cách 1: đăng nhập thủ công**

```bash
python fb_scraper.py --login
```

Đăng nhập trong cửa sổ Chrome vừa mở (xác minh 2 bước nếu có), rồi quay lại terminal và nhấn Enter.

**Cách 2: nạp cookie** (dùng khi cách 1 bị treo ở bước xác minh)

1. Trên Chrome thường đang đăng nhập Facebook, cài extension **Cookie-Editor**.
2. Mở facebook.com, vào Cookie-Editor, chọn **Export → JSON**.
3. Dán vào `cookies.json` (PowerShell: `Get-Clipboard | Set-Content cookies.json -Encoding utf8`).
4. Chạy `python fb_scraper.py --cookies cookies.json`, kiểm tra đã vào được Facebook rồi nhấn Enter.
5. **Xoá `cookies.json`.**

### 4.2. Chạy crawl

```bash
# Một bài viết
python fb_scraper.py "https://www.facebook.com/groups/xxxx/posts/123456/"

# N bài mới nhất
python fb_scraper.py "https://www.facebook.com/groups/uiter.confessions" --max-posts 50

# Toàn bộ nhóm (lưu dần, dừng giữa chừng rồi chạy tiếp được)
python fb_scraper.py "https://www.facebook.com/groups/uiter.confessions" --all

# Lấy thêm bài mới sau khi đã chạy --all
python fb_scraper.py "https://www.facebook.com/groups/uiter.confessions" --all --recollect
```

Trong lúc script chạy, **đừng thao tác trên cửa sổ Chrome**.

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `url` | | Link nhóm, trang hoặc bài viết |
| `--login` | | Mở trình duyệt để đăng nhập thủ công |
| `--cookies FILE` | | Nạp cookie từ file JSON (Cookie-Editor) |
| `--max-posts N` | 10 (`--all`: không giới hạn) | Số bài tối đa |
| `--max-scrolls N` | 2 × max-posts, tối thiểu 40 | Số lần cuộn feed tối đa |
| `--all` | | Lấy toàn bộ bài, lưu dần, chạy tiếp được |
| `--recollect` | | Dùng với `--all`: cuộn lại feed để tìm bài mới |
| `--rest-every N` | 30 | Dùng với `--all`: nghỉ 1–3 phút sau mỗi N bài |
| `--headless` | | Chạy ẩn trình duyệt (Facebook dễ chặn hơn, không khuyến khích) |

**Chế độ `--all`** làm 2 bước:
1. **Gom link:** sắp feed theo bài mới nhất, cuộn tới khi không còn bài mới, ghi link dần vào `links.txt`.
2. **Cào từng bài:** mỗi bài cào xong được ghi ngay vào `posts.jsonl`.

Khi bị dừng (Ctrl+C, mất mạng…), chỉ cần chạy lại đúng lệnh cũ: script bỏ qua các bài đã cào. Nếu Facebook bắt xác minh (checkpoint), script tự dừng; xử lý xong trên trình duyệt rồi chạy lại. Bài lỗi 2 lần liên tiếp được ghi vào `failed.txt` và cào lại ở lần sau.

### 4.3. Kết quả crawl

- **Chế độ thường:** `output/posts_<thời gian>.json` và `output/comments_<thời gian>.csv`.
- **Chế độ `--all`:** `output/<tên nhóm>/` gồm `links.txt`, `links.done`, `posts.jsonl`, `posts_all.json`, `comments_all.csv`, `failed.txt`.

Cấu trúc mỗi bài trong JSON:

```json
{
  "author": "Nguyễn Văn A",
  "text": "Cho mình hỏi gần khu KTX B có chỗ học TOEIC nào ổn không ạ?",
  "images": ["https://scontent..."],
  "url": "https://www.facebook.com/groups/xxxx/posts/123456/",
  "scraped_at": "2026-10-02T21:51:36",
  "comments": [
    {
      "author": "Trần Thị B",
      "text": "Mình đang học ở trung tâm X ...",
      "is_reply": false,
      "time": "29 phút",
      "link": "https://www.facebook.com/groups/xxxx/posts/123456/?comment_id=...",
      "aria_label": "Bình luận dưới tên Trần Thị B vào 29 phút trước"
    }
  ]
}
```

### 4.4. Hạn chế của scraper

- **Comment bị cào lặp:** cùng một comment có thể xuất hiện nhiều lần trong một bài. Các bước sau đã tự lọc phần lặp.
- **Lấy nhầm nội dung:** đôi khi lấy nhầm bài của trang khác được chia sẻ hoặc gắn vào post.
- **Thời gian tương đối:** thời gian comment lấy đúng như Facebook hiển thị ("1 giờ", "2 ngày"); bài viết chưa có trường thời gian.
- **Comment không có chữ:** comment chỉ có sticker hoặc ảnh sẽ có `text` rỗng.
- **Nhóm lớn:** Facebook có thể ngừng tải bài cũ sau một lúc cuộn (nhóm UIT Confessions chỉ gom được khoảng 290 link mỗi lần).
- **Giao diện thay đổi:** Facebook hay đổi cấu trúc trang. Khi không tìm được bài nào, script lưu `output/debug.png` và `output/debug.html` để kiểm tra.

---

## 5. Bước 2: Gộp dữ liệu vào final.json

Mỗi lần crawl tạo ra một file mới. `merge_json.py` gộp chúng vào **một file duy nhất `output/final.json`** và không tạo bản trùng.

```bash
python sentiment/merge_json.py output/posts_all.json
python sentiment/merge_json.py "output/*/posts.jsonl" output/posts_2026*.json
```

**Quy tắc gộp:**
- Bài trùng **link** (cùng `/posts/<id>`) hoặc trùng **nội dung** (bài đăng lại) với bài đã có thì không thêm lần nữa. Chỉ các **comment mới** của bài đó được bổ sung vào bài cũ (comment trùng = cùng người viết và cùng nội dung).
- Bài mới thì được thêm vào, đồng thời lọc comment bị cào lặp.
- Comment rỗng không được bổ sung.
- Trước khi ghi, script tự sao lưu `final_backup_<thời gian>.json`.

Gộp lại cùng một file nhiều lần cũng không tạo bản trùng.

---

## 6. Bước 3: Chuyển JSON sang CSV

```bash
python sentiment/json_to_csv.py                                       # mặc định đọc output/final.json
python sentiment/json_to_csv.py output/final.json --out sentiment/data/samples.csv
```

Mỗi **post** và mỗi **comment** thành **một dòng (một mẫu)**, chưa có nhãn:

| Cột | Ý nghĩa |
|---|---|
| `id` | Số thứ tự mẫu (được đánh lại mỗi lần chạy) |
| `source` | `post` hoặc `comment` |
| `text` | Nội dung đã làm sạch: chuẩn hoá Unicode NFC, xoá link, gộp khoảng trắng |
| `author` | Người viết |
| `is_reply`, `comment_time` | Chỉ có ở comment |
| `post_id` | ID của post chứa mẫu, dùng để nhóm comment theo post |
| `url` | Link post, hoặc link riêng của comment |

Script tự bỏ: post hoặc comment rỗng, post trùng giữa các file, và comment bị cào lặp trong cùng post. Số mẫu đã bỏ được in ra cuối.

Bộ dữ liệu hiện tại: **2.543 mẫu** (470 post, 2.073 comment).

> `predict.py` đọc được `.json` trực tiếp, nên có thể bỏ qua bước này nếu chỉ cần dự đoán.

---

## 7. Bước 4: Train model gốc trên VSMEC + VSFC

```bash
python sentiment/train_vsmec.py
```

Full fine-tune [`uitnlp/visobert`](https://huggingface.co/uitnlp/visobert) (BERT được pretrain trên mạng xã hội tiếng Việt, hiểu sẵn teencode và emoji) trên 2 bộ dữ liệu:

| Bộ dữ liệu | Nội dung | Train / Val / Test | Nhãn |
|---|---|---|---|
| [UIT-VSMEC](https://huggingface.co/datasets/tridm/UIT-VSMEC) | Bình luận Facebook | 5.548 / 686 / 693 | 7 cảm xúc, gộp thành 3 nhãn |
| [UIT-VSFC](https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback) | Phản hồi của sinh viên UIT về giảng viên, chương trình, cơ sở vật chất | 11.426 / 1.583 / 3.166 | Sẵn 3 nhãn |

**Gộp 7 cảm xúc của VSMEC thành 3 nhãn:**

| VSMEC | Nhãn |
|---|---|
| Enjoyment | positive |
| Sadness, Fear, Anger, Disgust | negative |
| Other | neutral |
| Surprise | neutral (hoặc bỏ, tuỳ `CONFIG["surprise"]`) |

**Cách train và đánh giá:**
- **Train** trên tập train của cả 2 bộ gộp lại.
- **Validation** dùng để chọn epoch tốt nhất và dừng sớm, theo **trung bình macro-F1 của từng bộ**, để VSFC (lớn và dễ hơn) không lấn át VSMEC.
- **Test** chỉ được đánh giá một lần cuối với model tốt nhất. Kết quả báo cáo riêng VSMEC, riêng VSFC và gộp chung, gồm: Accuracy, Precision/Recall/F1 (macro và weighted), MCC, Cohen's kappa, ROC-AUC, confusion matrix, độ chính xác theo cảm xúc gốc (VSMEC) và theo chủ đề (VSFC).

**Tham số chính** (sửa trong `CONFIG` ở đầu file):

| Tham số | Giá trị | Ghi chú |
|---|---|---|
| `datasets` | vsmec + vsfc, `train_fraction` 1.0 | Bật/tắt từng bộ, giảm tỉ lệ VSFC nếu nó lấn át |
| `surprise` | `"neutral"` | `"drop"` để bỏ các câu Surprise |
| `lowercase` | `False` | `True` để viết thường mọi câu (VSFC vốn viết thường hết) |
| `max_length` | 128 | 99% câu dưới 60 token |
| `epochs` / `early_stopping` | 20 / 5 | |
| `batch_size` | 32 | Giảm 16 nếu hết bộ nhớ GPU |
| `lr` / `head_lr` | 2e-5 / 1e-4 | Thân model / lớp phân loại |
| `dropout`, `label_smoothing` | 0,2 / 0,1 | Chống overfit |
| `class_weight` | `True` | Bù cho việc VSFC chỉ có ~5% neutral |

**Kết quả lưu ở** `sentiment/models/visobert_vsmec_vsfc/`: model, tokenizer, `metrics.json` (cấu hình, lịch sử từng epoch, metric val/test), `test_predictions.csv`.

> UIT-VSFC trên HuggingFace dùng script tải kiểu cũ mà `datasets` 4.x không còn hỗ trợ. Vì vậy script đọc bản Parquet do HuggingFace tự chuyển (`revision="refs/convert/parquet"`).

---

## 8. Bước 5: Gán nhãn dữ liệu UIT

Model gốc học trên dữ liệu ngoài, nên chưa quen văn phong confession: tâm sự, tìm người yêu, pass đồ, hỏi đáp, mỉa mai ":))". Vì vậy cần một tập câu UIT gán nhãn tay để fine-tune và đánh giá.

**File:** `sentiment/data/finetune.csv`, gồm 2 cột `text, label`.

| Nhãn | Comment | Post | Tổng |
|---|---|---|---|
| negative | 117 | 83 | 200 |
| neutral | 101 | 99 | 200 |
| positive | 147 | 15 | 162 |
| **Tổng** | 365 | 197 | **562** |

**Tiêu chí gán nhãn** (chỉ giữ câu có cảm xúc **rõ ràng**):

| Nhãn | Gồm | Ví dụ |
|---|---|---|
| **negative** | Than phiền, buồn, chán, cô đơn, lo lắng, chê bai, bức xúc, mỉa mai có ý tiêu cực rõ | "nhà xe bây giờ tắt ko chịu được", "Ai có gr đánh cầu ko cho mình xin vô đánh với ạ. Chán quá" |
| **positive** | Khen, cảm ơn, hào hứng, động viên | "Nhập form liền tayyy!!!", "t học chỗ thầy này t thấy ổn, dạy từ căn bản nên học dễ vào hơn" |
| **neutral** | Hỏi thông tin, trả lời thông tin, rao bán, thông báo | "mn cho em hỏi nếu mất thẻ sv thì làm lại ở đâu ạ", "4 kỹ năng nha em" |
| *bỏ qua* | Câu chỉ tag tên, câu vô nghĩa ("Chấm", "Bro"), cảm xúc lẫn lộn, mỉa mai khó đoán | "Nguyễn Văn A", "Tôi tạ quá" |

**Vì sao positive chưa đủ 200 câu:** trong 2.543 mẫu, câu có cảm xúc rõ ràng chỉ chiếm khoảng 10%. Phần lớn confession là hỏi đáp, tag tên bạn bè, tìm người yêu hoặc rủ chơi game. Câu khen hoặc cảm ơn rõ ràng còn hiếm hơn, nhất là ở post (chỉ có 15 post positive). Thay vì thêm câu mơ hồ cho đủ số, dự án giữ 162 câu positive và dùng class weight để bù lệch nhãn.

**Gợi ý để gán nhãn nhanh hơn:**
1. Chạy `predict.py` lên `samples.csv`.
2. Sắp theo `conf`, duyệt các câu model đoán positive/negative.
3. Sửa nhãn rồi chép vào `finetune.csv`.
4. Chạy `find_label_issues.py` (mục 11) để tìm các nhãn có thể bị gán sai.

---

## 9. Bước 6: Fine-tune LoRA

```bash
python sentiment/finetune_lora.py
```

- **Model gốc:** `models/visobert_vsmec_vsfc`.
- **Dữ liệu:** `data/finetune.csv`, chia train/test **80/20** theo tỉ lệ nhãn (`random_state=42`): 449 câu train, 113 câu test.
- **LoRA** gắn vào các lớp Linear trong encoder (query, key, value, các lớp dense). Lớp phân loại được train đầy đủ. Chỉ khoảng **2% tham số** (≈1,9 triệu) được train.
- **Tập test chỉ dùng để đánh giá cuối**, không dùng để chọn epoch, nên điểm không bị cao giả vì chọn theo test. Model train đúng số epoch cố định trong `CONFIG`.
- Script in **điểm trước và sau fine-tune trên cùng tập test**, cùng danh sách các câu test đổi kết quả dự đoán (✔ đổi đúng / ✘ đổi sai).

| Tham số | Giá trị | Ghi chú |
|---|---|---|
| `data_path` | `data/finetune.csv` | |
| `test_size` | 0,2 | |
| `lora_r` / `lora_alpha` / `lora_dropout` | 8 / 16 / 0,1 | Tăng `r` lên 16 nếu model học chưa đủ (underfit) |
| `epochs` | 15 | Train accuracy đạt 100% từ epoch 6; có thể giảm |
| `batch_size` | 16 | |
| `lr` | 2e-4 | LoRA cần lr cao hơn full fine-tune |
| `max_length` | 256 | Post confession dài hơn câu VSMEC/VSFC |

**Kết quả:**
- `models/visobert_uit_lora/`: chỉ adapter LoRA (~8MB).
- `models/visobert_uit/`: **model đã gộp LoRA**, được `predict.py` dùng mặc định. Thư mục này có thêm `metrics.json` (trước/sau), `test_predictions.csv` và `split.csv` (câu nào thuộc train/test).

---

## 10. Bước 7: Dự đoán

```bash
python sentiment/predict.py                                     # data/samples.csv
python sentiment/predict.py --input output/final.json           # đọc thẳng JSON
python sentiment/predict.py --input sentiment/data/finetune.csv
python sentiment/predict.py --text "Trường mình wifi lag quá trời" --text "Thầy dạy hay lắm"
python sentiment/predict.py --model sentiment/models/visobert_vsmec_vsfc
```

| Tham số | Ý nghĩa |
|---|---|
| `--input` | `.csv` (cần cột `text`) hoặc `.json`/`.jsonl` |
| `--out` | File kết quả; mặc định `<input>_predicted.csv` |
| `--model` | Thư mục model; mặc định dùng `visobert_uit`, nếu không có thì `visobert_vsmec_vsfc` |
| `--text` | Dự đoán câu nhập tay; dùng được nhiều lần |

**Kết quả:**

1. **`<input>_predicted.csv`:** giữ nguyên các cột cũ, thêm:
   - `pred`: nhãn dự đoán. Không ghi đè cột `label` nếu file đã có.
   - `conf`: độ tự tin (xác suất cao nhất).
   - `p_negative`, `p_neutral`, `p_positive`.
   - `uncertain`: `True` nếu `conf < 0,6`, nên xem lại.
2. **`<input>_predicted_posts.csv`** (khi đầu vào có post và comment): mỗi post một dòng, gồm nhãn của post, số comment, tỉ lệ % từng nhãn trong comment, và `comment_majority` là nhãn chiếm đa số trong comment (`mixed` nếu hoà).
3. **Màn hình:** phân bố nhãn theo post/comment, số câu không chắc, ví dụ tự tin nhất mỗi nhãn. Nếu file đầu vào có cột `label`, script in thêm bảng so sánh với nhãn đó.

Ví dụ (model `visobert_uit`):

```
Trường mình wifi lag quá trời     -> negative (conf=0.921)
Thầy dạy hay lắm, cảm ơn thầy     -> positive (conf=0.953)
Cho em hỏi phòng CTSV ở đâu ạ     -> neutral  (conf=0.941)
Nhà xe đông quá, 7h là hết chỗ    -> neutral  (conf=0.462)   <- conf thấp, đáng lẽ là negative
```

> Khi chạy `predict.py` trên `finetune.csv`, điểm sẽ **cao giả**, vì 80% số câu đã dùng để train. Điểm thật nằm trong `models/visobert_uit/metrics.json`.

---

## 11. Công cụ phụ: kiểm tra chất lượng nhãn

```bash
pip install cleanlab pyvi
python sentiment/find_label_issues.py
```

Dùng **Confident Learning** (thư viện cleanlab) để tìm các câu có thể bị gán nhãn sai:
1. Train 5-fold bằng 2 model **độc lập** với model đã gán nhãn nháp: TF-IDF + Logistic Regression, và embedding `BAAI/bge-m3` + Logistic Regression.
2. Lấy xác suất dự đoán out-of-fold: mỗi câu được dự đoán bởi một model chưa từng thấy nó.
3. Câu nào model rất tự tin là khác nhãn đang gán thì bị đánh dấu nghi sai. Thường đó là nhãn sai thật, hoặc câu mỉa mai.

> Trước khi chạy, đặt `CONFIG["data_path"]` trong `find_label_issues.py` trỏ tới file nhãn đang dùng (`data/finetune.csv`).

**Kết quả:** `data/label_issues.csv`. Các dòng `is_issue = True` nằm trên cùng, xếp theo `label_quality` tăng dần, kèm `suggested_label` (nhãn model đề xuất). Màn hình in tỉ lệ nghi sai theo nhãn, ma trận nhiễu ước lượng và 15 câu đáng nghi nhất.

Đây là danh sách **nên xem lại**, không phải để tự động sửa: model kiểm tra cũng có thể sai, nhất là với câu mỉa mai hoặc câu ngắn. Script này cần `common.py` và `pyvi`, đồng thời tải `BAAI/bge-m3` (khoảng 2,3GB) ở lần chạy đầu.

---

## 12. Báo cáo kết quả

Mọi số liệu dưới đây lấy từ lần train mới nhất (`metrics.json` và `test_predictions.csv` trong `sentiment/models/`), chạy trên 1 GPU NVIDIA GeForce RTX 3060 Laptop, seed 42.

### 12.1. Model gốc `visobert_vsmec_vsfc`

Train tối đa 20 epoch, dừng sớm ở epoch 11. **Model tốt nhất ở epoch 6** (trung bình macro-F1 validation = 0,792).

**Tập test** (đánh giá 1 lần với model tốt nhất):

| Tập test | Số câu | Accuracy | Macro-F1 | Weighted-F1 | MCC | ROC-AUC | F1 neg | F1 neu | F1 pos |
|---|---|---|---|---|---|---|---|---|---|
| VSMEC | 693 | 0,762 | 0,733 | 0,759 | 0,620 | 0,837 | 0,838 | 0,615 | 0,746 |
| VSFC | 3.166 | 0,926 | 0,821 | 0,925 | 0,865 | 0,923 | 0,943 | 0,573 | 0,947 |
| Gộp chung | 3.859 | 0,897 | 0,814 | 0,896 | 0,820 | 0,915 | 0,922 | 0,594 | 0,926 |

**Tập validation:**

| Tập val | Số câu | Accuracy | Macro-F1 | F1 neg | F1 neu | F1 pos |
|---|---|---|---|---|---|---|
| VSMEC | 686 | 0,765 | 0,746 | 0,825 | 0,616 | 0,798 |
| VSFC | 1.583 | 0,941 | 0,837 | 0,954 | 0,596 | 0,961 |
| Gộp chung | 2.269 | 0,888 | 0,818 | 0,914 | 0,610 | 0,928 |

**Confusion matrix trên test** (hàng = nhãn thật, cột = dự đoán):

| VSMEC | đoán neg | đoán neu | đoán pos |
|---|---|---|---|
| thật neg (334) | **293** | 31 | 10 |
| thật neu (166) | 42 | **100** | 24 |
| thật pos (193) | 30 | 28 | **135** |

| VSFC | đoán neg | đoán neu | đoán pos |
|---|---|---|---|
| thật neg (1.409) | **1.347** | 29 | 33 |
| thật neu (167) | 38 | **92** | 37 |
| thật pos (1.590) | 64 | 33 | **1.493** |

**Lịch sử train** (trung bình macro-F1 của 2 bộ trên validation):

| Epoch | 1 | 2 | 3 | 4 | 5 | **6** | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Train loss | 0,878 | 0,673 | 0,584 | 0,507 | 0,475 | **0,456** | 0,447 | 0,444 | 0,438 | 0,436 | 0,436 |
| Val TB macro-F1 | 0,699 | 0,753 | 0,782 | 0,788 | 0,786 | **0,792** | 0,789 | 0,784 | 0,785 | 0,780 | 0,781 |

**Nhận xét:**
- **neutral là nhãn yếu nhất** ở cả 2 bộ (F1 ~0,6). Ở VSFC, neutral chỉ chiếm ~5% và hay bị nhầm sang cả 2 nhãn còn lại. Ở VSMEC, neutral gộp từ Other và Surprise nên nội dung lẫn lộn.
- VSMEC (bình luận Facebook, văn phong tự do) khó hơn VSFC rất nhiều: accuracy 0,76 so với 0,93.
- Validation đạt đỉnh ở epoch 6 rồi giảm nhẹ trong khi train loss vẫn giảm, cho thấy model bắt đầu overfit. Dừng sớm đã chặn được việc này.

### 12.2. Model cuối `visobert_uit` (LoRA trên dữ liệu UIT)

Tập test gồm **113 câu UIT** (40 negative, 40 neutral, 33 positive), chưa từng dùng để train. Cùng tập test được dùng để chấm model trước và sau fine-tune.

| Metric | Trước fine-tune | **Sau fine-tune** | Chênh lệch |
|---|---|---|---|
| Accuracy | 0,673 | **0,938** | +0,265 |
| Macro-F1 | 0,665 | **0,938** | +0,273 |
| Weighted-F1 | 0,670 | **0,938** | +0,268 |
| MCC | 0,515 | **0,907** | +0,392 |
| Cohen's kappa | 0,501 | **0,907** | +0,406 |
| Macro ROC-AUC | 0,780 | **0,990** | +0,210 |
| F1 negative | 0,775 | **0,949** | +0,174 |
| F1 neutral | 0,632 | **0,925** | +0,293 |
| F1 positive | 0,588 | **0,941** | +0,353 |

**Confusion matrix** (hàng = nhãn thật, cột = dự đoán):

| Trước fine-tune | đoán neg | đoán neu | đoán pos |
|---|---|---|---|
| thật neg (40) | **31** | 9 | 0 |
| thật neu (40) | 7 | **30** | 3 |
| thật pos (33) | 2 | 16 | **15** |

| Sau fine-tune | đoán neg | đoán neu | đoán pos |
|---|---|---|---|
| thật neg (40) | **37** | 2 | 1 |
| thật neu (40) | 1 | **37** | 2 |
| thật pos (33) | 0 | 1 | **32** |

**Nhận xét:**
- **Model gốc bỏ sót positive kiểu confession:** trước fine-tune, 16/33 câu positive bị đoán thành neutral. Sau fine-tune chỉ còn 1 câu.
- 33 câu test đổi dự đoán sau fine-tune: **31 câu sửa sai thành đúng**, chỉ 1 câu từ đúng thành sai.
- Train accuracy đạt **100% từ epoch 6** và loss đứng yên ở ~0,297 (mức sàn do label smoothing). Model đã học thuộc tập train nhỏ; có thể giảm `epochs` xuống ~6–8 mà không mất điểm.

**7 câu test còn đoán sai:**

| Nhãn thật | Dự đoán | conf | Câu | Lý do có thể |
|---|---|---|---|---|
| negative | positive | 0,79 | "+1 2k8 Shock vì giá trọ gần trường" | Câu ngắn, "+1" thường đi với câu hào hứng |
| neutral | positive | 0,90 | "Tadv bạn sẽ dc thi 3 kỹ năng nghe đọc viết trước, điểm ổn thì sẽ có mail…" | Có từ "ổn" |
| neutral | negative | 0,50 | "bạn mình hôm qua … làm rơi cái ví … bên trong có hơn 1tr…" | Thông báo mất đồ, nội dung tiêu cực nhưng giọng trung tính |
| neutral | positive | 0,95 | "Xem trước chương trình đào tạo, học tập và chép đầy đủ, nghiêm chỉnh…" | Lời khuyên, giọng tích cực |
| negative | neutral | 0,92 | "có nhma luôn trong tình trạng kín b ạ" | Than phiền nhẹ, viết như câu trả lời thông tin |
| positive | neutral | 0,87 | "bạn tham khảo trung tâm DN English Academy thử xem…" | Lời khen lẫn trong câu giới thiệu |
| negative | neutral | 0,57 | "Mọi người ơi cho e hỏi phòng ở 4 người mà tháng trước…" | Bức xúc nằm trong câu hỏi |

Phần lớn lỗi nằm ở **ranh giới neutral với 2 nhãn còn lại**: câu hỏi chứa bức xúc, thông báo có nội dung tiêu cực, lời khuyên có giọng tích cực.

**So sánh các phương án đã thử** (macro-F1 trên câu UIT gán nhãn tay):

| Model | Macro-F1 |
|---|---|
| Chỉ train trên VSMEC | ~0,48 (gần như đoán neutral cho mọi câu) |
| VSMEC + VSFC | 0,665 |
| VSMEC + VSFC + LoRA trên 391 câu UIT (phiên bản dữ liệu cũ, test 79 câu) | 0,913 |
| **VSMEC + VSFC + LoRA trên 562 câu UIT (hiện tại, test 113 câu)** | **0,938** |

Hai dòng LoRA dùng tập test khác nhau nên không so sánh trực tiếp được, nhưng cả hai đều cho thấy fine-tune trên dữ liệu cùng miền là bước quan trọng nhất.

### 12.3. Phân tích cảm xúc UIT Confessions

Áp dụng model cuối lên toàn bộ **2.543 mẫu** (470 post, 2.073 comment):

| Nhãn dự đoán | Post | Comment | Tổng | Tỉ lệ |
|---|---|---|---|---|
| negative | 125 | 532 | 657 | 25,8% |
| neutral | 288 | 767 | 1.055 | 41,5% |
| positive | 57 | 774 | 831 | 32,7% |

- Độ tự tin trung bình 0,867. Có **180 mẫu (7,1%)** độ tự tin dưới 0,6, nên xem lại.
- **Post** chủ yếu là neutral (61%: hỏi đáp, rao bán, tìm bạn), negative đứng thứ hai (27%: than phiền, tâm sự). Post positive rất ít (12%).
- Theo **nhãn chiếm đa số trong comment** của từng post (338 post có comment): neutral 140, positive 100, negative 61, hoà (mixed) 37.

> ⚠️ **Tỉ lệ positive ở comment nhiều khả năng bị thổi phồng.** Rất nhiều comment chỉ là tag tên bạn bè hoặc "ứng cử", "ké", "lụm" dưới các bài tìm người yêu. Loại câu này đã bị loại khỏi tập gán nhãn (mục 8), nên model chưa từng học cách xử lý chúng và hay đoán là positive. Ví dụ bị đoán positive: "Duy Khải bạn tui nha", "Vĩnh Nguyên lụm", "Tuấn Phạm vào nhận liền". 37% câu bị đoán positive có độ dài ≤ 25 ký tự. Nên lọc riêng các comment chỉ tag tên trước khi thống kê.

### 12.4. Độ tin cậy của kết quả

- **Tập test UIT nhỏ (113 câu):** mỗi câu sai làm accuracy giảm ~0,9 điểm. Khoảng tin cậy 95% của accuracy 0,938 là khoảng **±4,5 điểm**.
- **Dữ liệu chỉ gồm câu "rõ ràng":** câu mơ hồ, câu tag tên, câu mỉa mai khó đoán đã bị loại khi gán nhãn. Điểm 0,938 phản ánh câu có cảm xúc rõ, **không phải** toàn bộ dữ liệu thật (xem 12.3).
- **Có thể có thiên lệch chọn mẫu:** một phần câu được chọn để gán nhãn sau khi xem dự đoán của model, nên có thể thiên về câu model dễ đoán. Mọi nhãn vẫn được người kiểm tra lại.

---

## 13. Lỗi thường gặp

| Lỗi | Nguyên nhân | Cách sửa |
|---|---|---|
| `ImportError: huggingface-hub>=1.5.0,<2.0 is required` | Gói khác đã nâng `huggingface-hub` lên 2.x | `pip install "huggingface-hub>=1.5,<2"` |
| `RuntimeError: Dataset scripts are no longer supported` | UIT-VSFC dùng script tải kiểu cũ | Đã xử lý trong `train_vsmec.py` (đọc bản Parquet) |
| Tokenizer cắt chữ sai ("rất" thành "▁r" + "ất") | `transformers` 5.x chuyển đổi sai file `sentencepiece.bpe.model` của ViSoBERT | Dùng `visobert_tokenizer.py` (đã dùng ở mọi script), vì nó gọi thẳng SentencePiece gốc |
| `UnicodeEncodeError ... cp1258` | Terminal Windows không in được tiếng Việt | `set PYTHONIOENCODING=utf-8` |
| `Server disconnected` / `Can't load the model` khi tải từ HuggingFace | Mạng chập chờn | Model đã có trong cache thì chạy offline: `set HF_HUB_OFFLINE=1` |
| `401 ... Repository Not Found ... models/visobert_uit` | Thư mục model không tồn tại, nên script hiểu nhầm là tên model trên HuggingFace | Kiểm tra đường dẫn hoặc train lại (Bước 4, 6) |
| `FileNotFoundError: ... fine_tune_v2.csv` | Script còn trỏ tới tên file nhãn cũ | Đổi `CONFIG["data_path"]` thành `data/finetune.csv` |
| `CUDA out of memory` | Batch quá lớn | Giảm `batch_size` (32 → 16 → 8) trong `CONFIG` |
| `ModuleNotFoundError: No module named 'common'` | Thiếu `common.py` | Chỉ `find_label_issues.py` cần file này; các script khác chạy độc lập |
| `the following arguments are required: inputs` | Chạy `merge_json.py` mà thiếu đường dẫn file | Thêm đường dẫn: `python sentiment/merge_json.py output/posts_all.json` |

---

## 14. Hạn chế và hướng phát triển

**Hạn chế:**
- **Ít dữ liệu UIT gán nhãn:** 562 câu, trong đó positive chỉ có 162 câu (15 post). Tập test 113 câu nên điểm có sai số khoảng ±4–5 điểm.
- **Câu chỉ tag tên, câu ngắn, câu mỉa mai** vẫn dễ bị đoán sai, và làm lệch thống kê trên dữ liệu thật (mục 12.3).
- **Nhãn neutral** là nhãn yếu nhất ở các bộ công khai, do Other/Surprise của VSMEC mơ hồ và VSFC chỉ có ~5% neutral.
- **Cảm xúc chung khác thái độ:** model đo cảm xúc chung của câu, không phải *thái độ đối với trường*. Ví dụ "Buồn quá mới chia tay" là negative dù không liên quan tới UIT.
- **Model học thuộc tập train:** LoRA đạt 100% train accuracy, nên khả năng tổng quát hoá phụ thuộc nhiều vào độ đa dạng của dữ liệu gán nhãn.

**Hướng phát triển:**
1. **Thêm nhãn hoặc bộ lọc "không liên quan"** cho comment chỉ tag tên / ứng cử, để thống kê trên dữ liệu thật chính xác hơn.
2. **Crawl thêm** các nhóm/trang khác về UIT hoặc confession sinh viên, rồi gán nhãn thêm, ưu tiên post positive.
3. **Đánh giá bằng k-fold** (5 lần, mỗi lần một phần 20% khác nhau làm test) để có điểm ổn định hơn với tập nhỏ.
4. **Gán nhãn nháp bằng LLM** theo đúng tiêu chí ở mục 8, rồi duyệt lại, để mở rộng dữ liệu nhanh hơn.
5. **Train VSMEC với 7 nhãn gốc** rồi mới gộp thành 3 nhãn, có thể giúp nhãn neutral.
6. **Thêm nhãn chủ đề** (giảng viên, cơ sở vật chất, học vụ…) để biết sinh viên than phiền về điều gì.

---

## 15. Lưu ý an toàn và quyền riêng tư

- **`fb_profile/` và `cookies.json` có giá trị như mật khẩu:** không chia sẻ, không đẩy lên git. Xoá `cookies.json` sau khi đăng nhập xong.
- Thu thập dữ liệu tự động vi phạm điều khoản của Facebook. Tài khoản có thể bị khoá tạm hoặc bị yêu cầu xác minh. Nên chia nhỏ mỗi buổi vài trăm bài, hoặc dùng tài khoản phụ.
- Chỉ thu thập dữ liệu bạn có quyền xem, dùng cho mục đích học tập hoặc nghiên cứu, và tôn trọng quyền riêng tư của người đăng. Dữ liệu chứa tên thật và nội dung cá nhân, nên **không công khai** `output/` hay các file CSV trong `sentiment/data/`.
- Model nặng ~390MB, vượt giới hạn 100MB/file của GitHub. Nếu muốn chia sẻ, đưa lên HuggingFace Hub hoặc dùng Git LFS; adapter LoRA (~8MB) thì đẩy được trực tiếp.

`.gitignore` đề xuất:

```
fb_profile/
cookies.json
output/
sentiment/data/
sentiment/models/
__pycache__/
```
