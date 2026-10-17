"""Dự đoán nhãn negative/neutral/positive bằng model ViSoBERT đã train.

    python sentiment/predict.py                                        # data/samples.csv
    python sentiment/predict.py --input ../output/final.json           # đọc thẳng JSON đã crawl
    python sentiment/predict.py --input data/fine_tune_v2.csv          # file có cột label -> in thêm điểm
    python sentiment/predict.py --text "Trường mình wifi lag quá trời" --text "Thầy dạy hay lắm"
    python sentiment/predict.py --model models/visobert_vsmec_vsfc     # dùng model khác

Đầu vào: .csv (cần cột text) hoặc .json/.jsonl (mỗi post và comment thành 1 mẫu, như json_to_csv.py).
Đầu ra (--out, mặc định <input>_predicted.csv): giữ nguyên các cột cũ, thêm
    pred, conf, p_negative, p_neutral, p_positive, uncertain (conf < ngưỡng)
Nếu có cột post_id: ghi thêm <out>_posts.csv - mỗi post 1 dòng: nhãn của post + tỉ lệ nhãn của comment.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification

from json_to_csv import clean_text, load_posts, to_rows
from visobert_tokenizer import ViSoBERTTokenizer

ROOT = Path(__file__).parent

# =============================================================================
# CẤU HÌNH
# =============================================================================
CONFIG = {
    # Model ưu tiên theo thứ tự; dùng model đầu tiên còn tồn tại
    "model_dirs": [ROOT / "models" / "visobert_uit",           # đã fine-tune LoRA trên confession UIT
                   ROOT / "models" / "visobert_vsmec_vsfc"],   # train trên VSMEC + VSFC
    "input": ROOT / "data" / "samples.csv",
    "max_length": 256,      # giống lúc fine-tune
    "batch_size": 64,
    "low_conf": 0.6,        # dưới ngưỡng này đánh dấu uncertain = True, nên xem lại
}
# =============================================================================


class Predictor:
    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = ViSoBERTTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir).to(self.device).eval()
        self.labels = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]

    @torch.no_grad()
    def predict_proba(self, texts, max_length=256, batch_size=64):
        """Trả về ma trận xác suất (n, số nhãn). Câu rỗng -> NaN."""
        texts = [clean_text(t) for t in texts]
        probs = np.full((len(texts), len(self.labels)), np.nan)
        idx = [i for i, t in enumerate(texts) if t]
        # Sắp theo độ dài để mỗi batch ít padding -> nhanh hơn
        idx.sort(key=lambda i: len(texts[i]))
        for s in range(0, len(idx), batch_size):
            chunk = idx[s:s + batch_size]
            enc = self.tok([texts[i] for i in chunk], max_length=max_length)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.device == "cuda"):
                logits = self.model(**enc).logits.float()
            probs[chunk] = logits.softmax(-1).cpu().numpy()
        return probs


def resolve_model(path):
    candidates = [Path(path)] if path else CONFIG["model_dirs"]
    for p in candidates:
        if (Path(p) / "config.json").exists():
            return Path(p)
    raise SystemExit(f"Không tìm thấy model trong: {[str(p) for p in candidates]}. Hãy train trước.")


def load_input(path):
    path = Path(path)
    if path.suffix in (".json", ".jsonl"):
        df, _ = to_rows(load_posts([str(path)]))
        return df
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "text" not in df:
        raise SystemExit(f"{path} không có cột 'text'")
    return df


def add_predictions(df, pred, cfg):
    probs = pred.predict_proba(df["text"].fillna("").astype(str).tolist(), cfg["max_length"], cfg["batch_size"])
    ok = ~np.isnan(probs).any(1)
    df["pred"] = pd.Series([pred.labels[i] for i in np.nan_to_num(probs).argmax(1)], index=df.index).where(ok)
    df["conf"] = pd.Series(np.nanmax(np.where(ok[:, None], probs, 0), 1).round(3), index=df.index).where(ok)
    for i, l in enumerate(pred.labels):
        df[f"p_{l}"] = probs[:, i].round(3)
    df["uncertain"] = (df["conf"] < cfg["low_conf"]).where(ok)
    return df


def post_summary(df, labels):
    """Mỗi post 1 dòng: nhãn của chính post + phân bố nhãn của comment bên dưới."""
    posts = df[df["source"] == "post"].set_index("post_id")[["text", "pred", "conf"]]
    posts.columns = ["post_text", "post_pred", "post_conf"]
    com = df[(df["source"] == "comment") & df["pred"].notna()]
    stats = pd.crosstab(com["post_id"], com["pred"]).reindex(columns=labels, fill_value=0)
    n = stats.sum(1)
    out = posts.join(n.rename("n_comments"), how="left").fillna({"n_comments": 0})
    out["n_comments"] = out["n_comments"].astype(int)
    for l in labels:
        out[f"comment_%{l}"] = (stats[l] / n).reindex(out.index).round(3)
    # Nhãn chiếm đa số trong comment; hoà nhau -> "mixed"
    tie = stats.eq(stats.max(1), axis=0).sum(1) > 1
    majority = stats.idxmax(1).where(~tie, "mixed")
    out["comment_majority"] = majority.reindex(out.index).where(out["n_comments"] > 0)
    return out.reset_index()


def main(cfg=CONFIG):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(cfg["input"]), help=".csv (cột text) hoặc .json/.jsonl")
    ap.add_argument("--out", help="mặc định: <input>_predicted.csv cạnh file input")
    ap.add_argument("--model", help="thư mục model (mặc định: model đầu tiên còn tồn tại trong CONFIG)")
    ap.add_argument("--text", action="append", help="dự đoán câu nhập tay (dùng được nhiều lần)")
    args = ap.parse_args()

    pred = Predictor(resolve_model(args.model))
    print(f"Model: {pred.model_dir} | thiết bị: {pred.device}")

    if args.text:
        probs = pred.predict_proba(args.text, cfg["max_length"])
        for t, p in zip(args.text, probs):
            print(f"\n{t}\n  -> {pred.labels[p.argmax()]} (conf={p.max():.3f})  " +
                  "  ".join(f"{l}={v:.3f}" for l, v in zip(pred.labels, p)))
        return

    df = add_predictions(load_input(args.input), pred, cfg)
    inp = Path(args.input)
    out = Path(args.out) if args.out else inp.with_name(f"{inp.stem}_predicted.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")

    valid = df[df["pred"].notna()]
    print(f"\n{len(df)} mẫu ({len(df) - len(valid)} câu rỗng bỏ qua). Phân bố nhãn dự đoán:")
    if "source" in df:
        table = pd.crosstab(valid["pred"], valid["source"], margins=True, margins_name="tổng")
        print(table.reindex(pred.labels + ["tổng"]).fillna(0).astype(int).to_string())
    else:
        print(valid["pred"].value_counts().reindex(pred.labels, fill_value=0).to_string())
    unc = int(valid["uncertain"].sum())
    print(f"\nĐộ tự tin trung bình: {valid['conf'].mean():.3f} | "
          f"không chắc (conf < {cfg['low_conf']}): {unc} mẫu ({unc / max(len(valid), 1):.1%})")

    # File có sẵn nhãn (vd fine_tune_v2.csv) -> so với nhãn đó
    if "label" in df and df["label"].isin(pred.labels).any():
        from sklearn.metrics import classification_report
        lab = valid[valid["label"].isin(pred.labels)]
        print(f"\nSo với cột label có sẵn ({len(lab)} câu) - lưu ý: câu đã dùng để train sẽ cho điểm cao giả:")
        print(classification_report(lab["label"], lab["pred"], labels=pred.labels, digits=3, zero_division=0))

    print("Ví dụ tự tin nhất mỗi nhãn:")
    for l in pred.labels:
        for _, r in valid[valid["pred"] == l].nlargest(3, "conf").iterrows():
            print(f"  [{l} {r['conf']:.2f}] {str(r['text'])[:110]}")
    print(f"\nĐã ghi {out}")

    if {"source", "post_id"} <= set(df.columns) and (df["source"] == "post").any():
        posts = post_summary(df, pred.labels)
        posts_out = out.with_name(f"{out.stem}_posts.csv")
        posts.to_csv(posts_out, index=False, encoding="utf-8-sig")
        print(f"Đã ghi {posts_out} ({len(posts)} post: nhãn post + tỉ lệ nhãn comment)")


if __name__ == "__main__":
    main()
