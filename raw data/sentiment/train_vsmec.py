"""Fine-tune uitnlp/visobert trên 2 bộ dữ liệu, 3 nhãn negative/neutral/positive:

  1. UIT-VSMEC (tridm/UIT-VSMEC): bình luận Facebook, 7 cảm xúc -> gộp 3 nhãn
        Enjoyment -> positive | Sadness, Fear, Anger, Disgust -> negative
        Other -> neutral      | Surprise -> neutral (hoặc bỏ, xem CONFIG)
  2. UIT-VSFC (uitnlp/vietnamese_students_feedback): phản hồi của sinh viên UIT, nhãn sẵn 3 lớp

Dùng đúng train / validation / test của từng bộ:
  - train: gộp train của 2 bộ
  - validation: chọn epoch tốt nhất + dừng sớm, theo TRUNG BÌNH macro-F1 của từng bộ
    (để VSFC - lớn và dễ hơn - không lấn át VSMEC)
  - test: đánh giá 1 lần cuối với model tốt nhất, báo cáo riêng từng bộ + gộp chung

Chỉnh tham số trong CONFIG rồi chạy:

    python sentiment/train_vsmec.py

Kết quả lưu ở CONFIG["output_dir"]: model, tokenizer, metrics.json, test_predictions.csv.
"""
import json
import random
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.metrics import (accuracy_score, classification_report, cohen_kappa_score,
                             confusion_matrix, matthews_corrcoef, precision_recall_fscore_support,
                             roc_auc_score)
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

from visobert_tokenizer import ViSoBERTTokenizer

ROOT = Path(__file__).parent
LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}

# =============================================================================
# CẤU HÌNH - sửa trực tiếp ở đây
# =============================================================================
CONFIG = {
    # ---- Dữ liệu
    "datasets": {
        # use: có dùng bộ này không | train_fraction: tỉ lệ tập train dùng (1.0 = toàn bộ)
        "vsmec": {"use": True, "train_fraction": 1.0},
        "vsfc": {"use": True, "train_fraction": 1.0},
    },
    "vsmec_emotion_map": {
        "Enjoyment": "positive",
        "Sadness": "negative", "Fear": "negative", "Anger": "negative", "Disgust": "negative",
        "Other": "neutral",
    },
    "surprise": "neutral",      # "neutral": gộp Surprise vào neutral | "drop": bỏ các câu Surprise
    "lowercase": False,         # True: viết thường mọi câu (VSFC vốn đã viết thường hết)
    "output_dir": ROOT / "models" / "visobert_vsmec_vsfc",

    # ---- Model
    "base_model": "uitnlp/visobert",
    "max_length": 128,          # 99% câu của 2 bộ < 60 token
    "dropout": 0.2,

    # ---- Huấn luyện
    "epochs": 20,
    "batch_size": 32,           # giảm 16 nếu hết bộ nhớ GPU
    "lr": 2e-5,                 # learning rate cho thân model
    "head_lr": 1e-4,            # learning rate cho lớp phân loại
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "max_grad_norm": 1.0,
    "class_weight": True,       # bù mất cân bằng nhãn (neutral rất ít trong VSFC)
    "label_smoothing": 0.1,
    "early_stopping": 5,        # dừng nếu điểm chọn model không tăng sau N epoch; 0 để tắt
    # Điểm chọn model trên validation: "mean_per_dataset" (TB macro-F1 từng bộ) | "combined" (gộp chung)
    "select_by": "mean_per_dataset",
    "fp16": True,               # chỉ có tác dụng khi có GPU
    "seed": 42,
}
# =============================================================================

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_SPACE_RE = re.compile(r"\s+")
_SPACE_PUNCT_RE = re.compile(r"\s+([.,!?;:)])")
SPLITS = [("train", "train"), ("val", "validation"), ("test", "test")]
VSFC_TOPICS = ["lecturer", "training_program", "facility", "others"]


def clean_text(text, lowercase=False):
    text = unicodedata.normalize("NFC", text or "")
    text = _SPACE_RE.sub(" ", _URL_RE.sub(" ", text)).strip()
    return text.lower() if lowercase else text


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------- Đọc dữ liệu
def load_vsmec(cfg):
    emap = dict(cfg["vsmec_emotion_map"])
    if cfg["surprise"] == "neutral":
        emap["Surprise"] = "neutral"
    ds = load_dataset("tridm/UIT-VSMEC")
    out = {}
    for name, key in SPLITS:
        df = ds[key].to_pandas().rename(columns={"Sentence": "text", "Emotion": "group"})
        df = df[df["group"].isin(emap)]
        df["label"] = df["group"].map(emap)
        out[name] = df
    return out


def load_vsfc(cfg):
    # Repo gốc dùng script tải (datasets 4.x không còn hỗ trợ) -> đọc bản parquet HF chuyển sẵn
    ds = load_dataset("uitnlp/vietnamese_students_feedback", revision="refs/convert/parquet")
    out = {}
    for name, key in SPLITS:
        df = ds[key].to_pandas().rename(columns={"sentence": "text"})
        df["label"] = df["sentiment"].map(dict(enumerate(LABELS)))  # 0 neg, 1 neu, 2 pos
        df["group"] = df["topic"].map(dict(enumerate(VSFC_TOPICS)))
        # VSFC đã tách token: "giáo trình chưa cụ thể ." -> "giáo trình chưa cụ thể."
        df["text"] = df["text"].str.replace(_SPACE_PUNCT_RE, r"\1", regex=True)
        out[name] = df
    return out


def load_data(cfg):
    loaders = {"vsmec": load_vsmec, "vsfc": load_vsfc}
    splits = {name: [] for name, _ in SPLITS}
    for ds_name, opt in cfg["datasets"].items():
        if not opt["use"]:
            continue
        for split, df in loaders[ds_name](cfg).items():
            df = df.assign(dataset=ds_name)
            df["text"] = df["text"].map(lambda t: clean_text(t, cfg["lowercase"]))
            df = df[df["text"] != ""]
            if split == "train" and opt["train_fraction"] < 1.0:
                df = df.sample(frac=opt["train_fraction"], random_state=cfg["seed"])
            splits[split].append(df[["dataset", "text", "label", "group"]])
    splits = {k: pd.concat(v, ignore_index=True) for k, v in splits.items()}

    print("Số câu theo nhãn (sau khi gộp nhãn):")
    rows = {}
    for split, df in splits.items():
        for ds_name, d in df.groupby("dataset"):
            rows[f"{split}/{ds_name}"] = d["label"].value_counts()
        rows[f"{split}/TỔNG"] = df["label"].value_counts()
    table = pd.DataFrame(rows).reindex(LABELS).fillna(0).astype(int)
    table.loc["tổng"] = table.sum()
    print(table.to_string(), "\n")
    return splits


# ---------------------------------------------------------------- Train / đánh giá
def make_loader(df, tok, cfg, shuffle):
    data = list(zip(df["text"], df["label"].map(LABEL2ID)))

    def collate(batch):
        xs, ys = zip(*batch)
        enc = tok(list(xs), max_length=cfg["max_length"])
        enc["labels"] = torch.tensor(ys)
        return enc
    return DataLoader(data, batch_size=cfg["batch_size"], shuffle=shuffle, collate_fn=collate)


@torch.no_grad()
def predict_proba(model, loader, device, use_fp16):
    model.eval()
    probs, losses = [], []
    for b in loader:
        b = {k: v.to(device) for k, v in b.items()}
        labels = b.pop("labels")
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_fp16):
            logits = model(**b).logits.float()
        losses.append(torch.nn.functional.cross_entropy(logits, labels, reduction="none").cpu().numpy())
        probs.append(logits.softmax(-1).cpu().numpy())
    return np.vstack(probs), np.concatenate(losses)


def compute_metrics(y_true, probs, losses=None):
    y_pred = probs.argmax(1)
    m = {"n": len(y_true), "accuracy": accuracy_score(y_true, y_pred)}
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average=avg, zero_division=0)
        m.update({f"{avg}_precision": p, f"{avg}_recall": r, f"{avg}_f1": f})
    m["mcc"] = matthews_corrcoef(y_true, y_pred)
    m["cohen_kappa"] = cohen_kappa_score(y_true, y_pred)
    try:
        m["macro_roc_auc_ovr"] = roc_auc_score(y_true, probs, multi_class="ovr", average="macro",
                                               labels=range(len(LABELS)))
    except ValueError:
        pass
    _, _, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=range(len(LABELS)), zero_division=0)
    m.update({f"f1_{l}": f[i] for i, l in enumerate(LABELS)})
    if losses is not None:
        m["loss"] = float(losses.mean())
    return {k: (v if k == "n" else round(float(v), 4)) for k, v in m.items()}


def eval_split(df, probs, losses):
    """Metric cho từng bộ dữ liệu + gộp chung, và điểm dùng để chọn model."""
    y = df["label"].map(LABEL2ID).to_numpy()
    res = {"combined": compute_metrics(y, probs, losses)}
    for ds_name in df["dataset"].unique():
        mask = (df["dataset"] == ds_name).to_numpy()
        res[ds_name] = compute_metrics(y[mask], probs[mask], losses[mask])
    per_ds = [v["macro_f1"] for k, v in res.items() if k != "combined"]
    res["mean_macro_f1"] = round(float(np.mean(per_ds)), 4)
    return res


def print_report(title, df, probs, losses):
    y_true = df["label"].map(LABEL2ID).to_numpy()
    y_pred = probs.argmax(1)
    m = compute_metrics(y_true, probs, losses)
    print(f"\n{'=' * 74}\n{title} ({len(df)} câu)\n{'=' * 74}")
    print(f"Accuracy          : {m['accuracy']:.4f}")
    print(f"Macro    P/R/F1   : {m['macro_precision']:.4f} / {m['macro_recall']:.4f} / {m['macro_f1']:.4f}")
    print(f"Weighted P/R/F1   : {m['weighted_precision']:.4f} / {m['weighted_recall']:.4f} / {m['weighted_f1']:.4f}")
    print(f"MCC               : {m['mcc']:.4f}")
    print(f"Cohen's kappa     : {m['cohen_kappa']:.4f}")
    if "macro_roc_auc_ovr" in m:
        print(f"Macro ROC-AUC(OvR): {m['macro_roc_auc_ovr']:.4f}")
    print(f"Loss              : {m['loss']:.4f}\n")
    print(classification_report(y_true, y_pred, labels=range(len(LABELS)), target_names=LABELS,
                                digits=4, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=range(len(LABELS)))
    print("Confusion matrix (hàng = thật, cột = dự đoán):")
    print(pd.DataFrame(cm, index=[f"thật_{l}" for l in LABELS], columns=[f"đoán_{l}" for l in LABELS]).to_string())

    names = {"vsmec": "cảm xúc gốc", "vsfc": "chủ đề"}
    for ds_name, d in df.assign(correct=(y_pred == y_true), pred=[LABELS[i] for i in y_pred]).groupby("dataset"):
        print(f"\nĐộ chính xác theo {names.get(ds_name, 'nhóm')} ({ds_name}):")
        s = d.groupby("group").agg(n=("correct", "size"), acc=("correct", "mean"),
                                   nhãn_thật_chính=("label", lambda x: Counter(x).most_common(1)[0][0]),
                                   hay_đoán_thành=("pred", lambda x: Counter(x).most_common(1)[0][0]))
        s["acc"] = s["acc"].map("{:.1%}".format)
        print(s.sort_values("n", ascending=False).to_string())
    return m


def main(cfg=CONFIG):
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = cfg["fp16"] and device == "cuda"
    print(f"Thiết bị: {device}" + (" (fp16)" if use_fp16 else ""))

    data = load_data(cfg)
    tok = ViSoBERTTokenizer.from_pretrained(cfg["base_model"])
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["base_model"], num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=LABEL2ID,
        classifier_dropout=cfg["dropout"]).to(device)

    train_dl = make_loader(data["train"], tok, cfg, shuffle=True)
    val_dl = make_loader(data["val"], tok, cfg, shuffle=False)
    test_dl = make_loader(data["test"], tok, cfg, shuffle=False)
    y_train = data["train"]["label"].map(LABEL2ID).to_numpy()

    weight = None
    if cfg["class_weight"]:
        counts = np.bincount(y_train, minlength=len(LABELS)).clip(min=1)
        weight = torch.tensor(len(y_train) / (len(LABELS) * counts), dtype=torch.float, device=device)
        print("Trọng số lớp:", {l: round(w, 2) for l, w in zip(LABELS, weight.tolist())})
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight, label_smoothing=cfg["label_smoothing"])

    no_decay = ("bias", "LayerNorm.weight")
    groups = []
    for is_head in (False, True):
        for decay in (True, False):
            params = [p for n, p in model.named_parameters()
                      if n.startswith("classifier") == is_head and (not any(nd in n for nd in no_decay)) == decay]
            groups.append({"params": params, "lr": cfg["head_lr"] if is_head else cfg["lr"],
                           "weight_decay": cfg["weight_decay"] if decay else 0.0})
    opt = torch.optim.AdamW(groups)
    steps = len(train_dl) * cfg["epochs"]
    sched = get_linear_schedule_with_warmup(opt, int(cfg["warmup_ratio"] * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    ds_names = list(data["val"]["dataset"].unique())
    out_dir = Path(cfg["output_dir"])
    history, best_score, best_epoch, bad_epochs = [], -1.0, 0, 0
    select = "TB macro-F1 từng bộ" if cfg["select_by"] == "mean_per_dataset" else "macro-F1 gộp"
    print(f"\nChọn model theo: {select} trên validation")
    header = f"{'epoch':>5} {'train_loss':>10} {'val_loss':>9} " + \
             "".join(f"{'F1_' + d:>10}" for d in ds_names) + \
             f" {'TB_F1':>7} {'gộp_F1':>7} {'F1_neg':>7} {'F1_neu':>7} {'F1_pos':>7} {'time':>6}"
    print(header)
    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        total, t0 = 0.0, time.time()
        for b in train_dl:
            b = {k: v.to(device) for k, v in b.items()}
            labels = b.pop("labels")
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_fp16):
                loss = loss_fn(model(**b).logits.float(), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
            scaler.step(opt)
            scaler.update()
            sched.step()
            opt.zero_grad()
            total += loss.item()

        probs, losses = predict_proba(model, val_dl, device, use_fp16)
        r = eval_split(data["val"], probs, losses)
        score = r["mean_macro_f1"] if cfg["select_by"] == "mean_per_dataset" else r["combined"]["macro_f1"]
        history.append({"epoch": ep, "train_loss": round(total / len(train_dl), 4), "val": r})
        mark = ""
        if score > best_score:
            best_score, best_epoch, bad_epochs = score, ep, 0
            model.save_pretrained(out_dir)
            tok.save_pretrained(out_dir)
            mark = "  *"
        else:
            bad_epochs += 1
        c = r["combined"]
        print(f"{ep:>5} {total / len(train_dl):>10.4f} {c['loss']:>9.4f} " +
              "".join(f"{r[d]['macro_f1']:>10.4f}" for d in ds_names) +
              f" {r['mean_macro_f1']:>7.4f} {c['macro_f1']:>7.4f} {c['f1_negative']:>7.4f} "
              f"{c['f1_neutral']:>7.4f} {c['f1_positive']:>7.4f} {time.time() - t0:>5.0f}s{mark}")
        if cfg["early_stopping"] and bad_epochs >= cfg["early_stopping"]:
            print(f"Dừng sớm: {bad_epochs} epoch liên tiếp không cải thiện.")
            break
    print(f"(* = model tốt nhất theo {select}, đã lưu) -> epoch {best_epoch}")

    # ---------------------------------------------------------------- Đánh giá model tốt nhất
    model = AutoModelForSequenceClassification.from_pretrained(out_dir).to(device)
    results, all_test = {}, None
    for split, loader in [("val", val_dl), ("test", test_dl)]:
        df = data[split]
        probs, losses = predict_proba(model, loader, device, use_fp16)
        split_name = "VALIDATION" if split == "val" else "TEST"
        results[split] = {}
        for ds_name in ds_names:
            mask = (df["dataset"] == ds_name).to_numpy()
            results[split][ds_name] = print_report(f"{split_name} / {ds_name.upper()} - model tốt nhất (epoch {best_epoch})",
                                                   df[mask], probs[mask], losses[mask])
        if len(ds_names) > 1:
            results[split]["combined"] = print_report(f"{split_name} / GỘP CHUNG - model tốt nhất (epoch {best_epoch})",
                                                      df, probs, losses)
        if split == "test":
            all_test = df.assign(pred=[LABELS[i] for i in probs.argmax(1)], conf=probs.max(1).round(3),
                                 **{f"p_{l}": probs[:, i].round(3) for i, l in enumerate(LABELS)})

    # ---------------------------------------------------------------- Bảng tóm tắt
    print(f"\n{'=' * 74}\nTÓM TẮT (model tốt nhất, epoch {best_epoch})\n{'=' * 74}")
    keys = ["n", "accuracy", "macro_f1", "weighted_f1", "mcc", "macro_roc_auc_ovr",
            "f1_negative", "f1_neutral", "f1_positive"]
    summary = pd.DataFrame({f"{s}/{d}": {k: m.get(k) for k in keys}
                            for s, by_ds in results.items() for d, m in by_ds.items()}).T
    print(summary.to_string())

    all_test.to_csv(out_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    meta = {"config": {k: str(v) if isinstance(v, Path) else v for k, v in cfg.items()},
            "best_epoch": best_epoch, "history": history, **results}
    (out_dir / "metrics.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nĐã lưu model + metrics.json + test_predictions.csv vào {out_dir}")


if __name__ == "__main__":
    main()
