"""Kiểm tra chất lượng nhãn trong data/labels.csv bằng Confident Learning (cleanlab).

Ý tưởng: train k-fold, lấy xác suất dự đoán out-of-fold (mỗi mẫu được dự đoán bởi model
CHƯA từng thấy nó). Mẫu nào model rất tự tin là khác nhãn gốc -> nghi nhãn sai / câu mỉa mai.

Model dùng để kiểm tra phải ĐỘC LẬP với model đã gán nhãn nháp (wonrax trong prepare.py),
nếu không nó chỉ "đồng ý với chính mình". Mặc định gộp xác suất của 2 model:
  - TF-IDF (từ + ký tự) + Logistic Regression
  - Embedding BAAI/bge-m3 (đa ngôn ngữ, mạnh tiếng Việt) + Logistic Regression

    python sentiment/find_label_issues.py

Kết quả: data/label_issues.csv - toàn bộ dòng, các dòng nghi sai nằm trên cùng, kèm
`suggested_label` (nhãn model đề xuất) và `label_quality` (0..1, càng thấp càng đáng nghi).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from cleanlab.count import estimate_joint
from cleanlab.dataset import overall_label_health_score
from cleanlab.filter import find_label_issues
from cleanlab.rank import get_label_quality_scores
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from common import LABEL2ID, LABELS, clean_text, normalize_vi, segment

ROOT = Path(__file__).parent

# =============================================================================
# CẤU HÌNH
# =============================================================================
CONFIG = {
    "data_path": ROOT / "data" / "fine_tune.csv",
    "out_path": ROOT / "data" / "label_issues.csv",
    "models": ["tfidf", "embedding"],   # bỏ bớt 1 trong 2 nếu muốn
    "embed_model": "BAAI/bge-m3",       # hoặc "vinai/phobert-base" (cần tải ~540MB)
    "embed_batch_size": 32,
    "embed_max_length": 256,
    "n_folds": 5,
    # Cách cleanlab lọc mẫu nghi sai: "prune_by_noise_rate" (mặc định cleanlab),
    # "both" (khắt khe hơn, ít báo nhầm hơn), "confident_learning"
    "filter_by": "prune_by_noise_rate",
    "seed": 42,
}
# =============================================================================


def tfidf_features(texts):
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    seg = [segment(normalize_vi(t)).lower() for t in texts]
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit_transform(seg)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True).fit_transform(seg)
    return hstack([word, char]).tocsr()


def embed_features(texts, cfg):
    import torch
    from transformers import AutoModel, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = cfg["embed_model"]
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name).to(device).eval()
    if device == "cuda":
        model = model.half()
    # PhoBERT cần input đã tách từ; bge-m3 (SentencePiece) thì không
    prep = (lambda t: segment(normalize_vi(t))) if "phobert" in name.lower() else normalize_vi
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), cfg["embed_batch_size"]):
            batch = [prep(t) for t in texts[i:i + cfg["embed_batch_size"]]]
            enc = tok(batch, padding=True, truncation=True, max_length=cfg["embed_max_length"],
                      return_tensors="pt").to(device)
            h = model(**enc).last_hidden_state
            if "bge" in name.lower():
                v = h[:, 0]  # bge dùng vector [CLS]
            else:
                m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
                v = (h * m).sum(1) / m.sum(1)  # mean pooling
            out.append(torch.nn.functional.normalize(v.float(), dim=-1).cpu().numpy())
    return np.vstack(out)


def oof_probs(X, y, cfg, C):
    """Xác suất out-of-fold: mỗi mẫu được dự đoán bởi model train trên các fold còn lại."""
    probs = np.zeros((len(y), len(LABELS)))
    cv = StratifiedKFold(n_splits=cfg["n_folds"], shuffle=True, random_state=cfg["seed"])
    for tr, te in cv.split(X, y):
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000)
        clf.fit(X[tr], y[tr])
        probs[te] = clf.predict_proba(X[te])
    return probs


def main(cfg=CONFIG):
    df = pd.read_csv(cfg["data_path"], encoding="utf-8-sig")
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["label"].isin(LABELS)].reset_index(drop=True)
    if "source" not in df:
        df["source"] = "post"
    texts = df["text"].map(clean_text).tolist()
    y = df["label"].map(LABEL2ID).to_numpy()
    print(f"{len(df)} mẫu, {cfg['n_folds']}-fold, model: {cfg['models']}\n")

    all_probs = []
    for name in cfg["models"]:
        if name == "tfidf":
            X, C = tfidf_features(texts), 5.0
        else:
            print(f"Tính embedding bằng {cfg['embed_model']}...")
            X, C = embed_features(texts, cfg), 2.0
        p = oof_probs(X, y, cfg, C)
        acc = (p.argmax(1) == y).mean()
        print(f"[{name}] độ khớp out-of-fold với nhãn hiện tại: {acc:.1%}")
        all_probs.append(p)
    probs = np.mean(all_probs, axis=0)
    print(f"[gộp] độ khớp out-of-fold với nhãn hiện tại: {(probs.argmax(1) == y).mean():.1%}\n")

    issues = find_label_issues(y, probs, filter_by=cfg["filter_by"])
    quality = get_label_quality_scores(y, probs, method="self_confidence")

    df["suggested_label"] = [LABELS[i] for i in probs.argmax(1)]
    df["label_quality"] = quality.round(3)
    df["is_issue"] = issues
    for i, l in enumerate(LABELS):
        df[f"p_{l}"] = probs[:, i].round(3)
    front = ["is_issue", "label_quality", "label", "suggested_label", "source", "text"]
    df = df[front + [c for c in df.columns if c not in front]]
    df = df.sort_values(["is_issue", "label_quality"], ascending=[False, True])
    df.to_csv(cfg["out_path"], index=False, encoding="utf-8-sig")

    # ---------------------------------------------------------------- Báo cáo
    n = int(issues.sum())
    print(f"== Nghi nhãn sai: {n}/{len(df)} mẫu ({n / len(df):.1%}) ==")
    print(f"Điểm sức khoẻ nhãn tổng thể (cleanlab, 0..1): "
          f"{overall_label_health_score(y, probs, verbose=False):.3f}\n")
    bad = df[df["is_issue"]]
    print("Theo nguồn:", bad["source"].value_counts().to_dict())
    print("\nNhãn hiện tại (hàng) -> nhãn model đề xuất (cột), chỉ các mẫu nghi sai:")
    print(pd.crosstab(bad["label"], bad["suggested_label"]).reindex(index=LABELS, columns=LABELS, fill_value=0))
    print("\nTỉ lệ nghi sai theo từng nhãn:")
    rate = df.groupby("label")["is_issue"].mean().reindex(LABELS)
    print(rate.map("{:.1%}".format).to_string())

    joint = estimate_joint(y, probs)
    print("\nMa trận ước lượng P(nhãn hiện tại, nhãn thật) - đường chéo càng lớn càng sạch:")
    print(pd.DataFrame(joint, index=[f"gán={l}" for l in LABELS],
                       columns=[f"thật={l}" for l in LABELS]).round(3))

    print("\n15 mẫu đáng nghi nhất:")
    for _, r in bad.head(15).iterrows():
        print(f"  [{r['label']} -> {r['suggested_label']} | q={r['label_quality']:.2f} | {r['source']}] "
              f"{r['text'][:110]}")
    print(f"\nĐã ghi {cfg['out_path']}")


if __name__ == "__main__":
    main()
