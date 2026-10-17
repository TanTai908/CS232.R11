"""Hàm dùng chung: đọc post, làm sạch text, tách từ, nhận diện post liên quan UIT."""
import glob
import json
import re
import unicodedata

from pyvi import ViTokenizer

LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}

# Từ khóa cho biết post có nhắc tới trường (cơ sở vật chất, đào tạo, giảng viên, thủ tục...)
UIT_KEYWORDS = [
    "uit", "trường", "khoa", "thầy", "cô ", "giảng viên", "gv", "môn", "học phí",
    "phòng đào tạo", "pđt", "ctsv", "ktx", "ký túc", "thư viện", "giảng đường",
    "điểm rèn luyện", "drl", "tín chỉ", "đăng ký học phần", "dkhp", "học bổng",
    "tốt nghiệp", "đầu ra", "chương trình", "lớp", "đồ án", "thi cuối kỳ", "giữa kỳ",
    "nhà xe", "bãi xe", "canteen", "căn tin", "wifi", "cơ sở", "đhqg", "làng đại học",
]

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = _URL_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


def segment(text: str) -> str:
    """Tách từ tiếng Việt (PhoBERT yêu cầu input đã tách từ, vd: 'sinh_viên')."""
    return ViTokenizer.tokenize(text)


# ---------------------------------------------------------------- Chuẩn hoá tiếng Việt MXH cho PhoBERT
# Teencode / viết tắt hay gặp trên Facebook -> từ chuẩn (chỉ thay khi đứng riêng thành 1 từ).
# Bỏ qua các từ mơ hồ như "a", "h", "bt", "k" đứng sau số (50k = 50 nghìn).
TEENCODE = {
    "ko": "không", "k": "không", "kh": "không", "khg": "không", "hok": "không", "hong": "không",
    "khum": "không", "kp": "không phải", "kô": "không",
    "dc": "được", "đc": "được",
    "j": "gì", "z": "vậy", "v": "vậy", "r": "rồi", "ròi": "rồi",
    "mn": "mọi người", "mng": "mọi người", "vs": "với", "ntn": "như thế nào",
    "cx": "cũng", "cg": "cũng", "ms": "mới", "lm": "làm", "ns": "nói", "trc": "trước",
    "bh": "bây giờ", "bjo": "bây giờ", "nma": "nhưng mà", "nhma": "nhưng mà",
    "hc": "học", "sv": "sinh viên", "gv": "giảng viên", "e": "em",
    "tks": "cảm ơn", "thanks": "cảm ơn", "thank": "cảm ơn", "camon": "cảm ơn",
    "dth": "dễ thương", "iu": "yêu", "thik": "thích", "bik": "biết", "bít": "biết",
}
# Emoji mang cảm xúc -> từ tiếng Việt (vocab PhoBERT gần như không có emoji)
EMOJI = {
    "vui": "😀😃😄😁😆😊☺🙂😉🥰😍😘🤗🥳",
    "cười": "😂🤣😹",
    "thích": "❤♥💕💖💗💓💞👍👏🙌💯🔥",
    "buồn": "😢😭😞😔😟🥺😿💔",
    "tức giận": "😡😠🤬👎💢",
    "chán": "😒🙄😑😐😤😩😫",
    "sợ": "😨😰😱",
}
_EMOJI_RE = {w: re.compile("[" + e + "]+️?") for w, e in EMOJI.items()}
_TEENCODE_RE = re.compile(r"(?<![\w\d])(" + "|".join(sorted(map(re.escape, TEENCODE), key=len, reverse=True))
                          + r")(?![\w])", re.IGNORECASE)
_REPEAT_RE = re.compile(r"(\w)\1{2,}")  # "quáaaaa" -> "quáa": tiếng Việt không có chữ lặp 3 lần
_TAIL_VOWEL_RE = re.compile(r"(\w)([aeiouy])\2*", re.IGNORECASE)


def _drop_tail_vowel(m):
    """'quáa' -> 'quá': bỏ nguyên âm không dấu kéo dài sau nguyên âm có dấu cùng gốc."""
    accented, plain = m[1], m[2]
    base = unicodedata.normalize("NFD", accented)[0].lower()
    return accented if accented.lower() != plain.lower() and base == plain.lower() else m[0]


def normalize_vi(text: str) -> str:
    text = clean_text(text)
    for word, pattern in _EMOJI_RE.items():
        text = pattern.sub(f" {word} ", text)
    text = _REPEAT_RE.sub(r"\1", text)
    text = _TAIL_VOWEL_RE.sub(_drop_tail_vowel, text)
    text = _TEENCODE_RE.sub(lambda m: TEENCODE[m[1].lower()], text)
    return _SPACE_RE.sub(" ", text).strip()


def preprocess(text: str) -> str:
    """Pipeline đầu vào PhoBERT, dùng chung cho train và predict: chuẩn hoá -> tách từ."""
    return segment(normalize_vi(text))


def load_posts(patterns):
    """Đọc post từ các file/glob: posts_*.json, posts_all.json (list) hoặc posts.jsonl (mỗi dòng 1 bài)."""
    posts = []
    for pat in patterns:
        files = sorted(glob.glob(pat))
        if not files:
            print(f"Không tìm thấy file: {pat}")
        for f in files:
            with open(f, encoding="utf-8") as fh:
                if f.endswith(".jsonl"):
                    posts += [json.loads(line) for line in fh if line.strip()]
                else:
                    posts += json.load(fh)
    return posts


def is_uit_related(text: str) -> bool:
    t = f" {text.lower()} "
    return any(k in t for k in UIT_KEYWORDS)
