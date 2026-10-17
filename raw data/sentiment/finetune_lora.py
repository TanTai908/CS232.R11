"""Fine-tune LoRA model ViSoBERT (đã train trên VSMEC + VSFC) với dữ liệu confession UIT đã gán nhãn tay.

    python sentiment/finetune_lora.py

Dữ liệu: data/fine_tune.csv (cột text, label), chia train/test 80/20 phân tầng theo nhãn.
  - Test CHỈ dùng để đánh giá cuối (không chọn epoch theo test -> điểm không bị cao giả).
  - Đánh giá model gốc TRƯỚC fine-tune trên cùng tập test để so sánh.

Kết quả:
  - CONFIG["adapter_dir"]: chỉ adapter LoRA (vài MB)
  - CONFIG["merged_dir"] : model đã gộp LoRA, dùng trực tiếp với predict.py (đổi model_dir)
  - metrics.json, test_predictions.csv, split.csv (câu nào thuộc train/test) trong merged_dir
"""
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from sklearn.metrics import (accuracy_score, classification_report, cohen_kappa_score,
                             confusion_matrix, matthews_corrcoef, precision_recall_fscore_support,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

from visobert_tokenizer import ViSoBERTTokenizer

ROOT = Path(__file__).parent

# =============================================================================
# CẤU HÌNH - sửa trực tiếp ở đây
# =============================================================================
CONFIG = {
    # ---- Dữ liệu
    "data_path": ROOT / "data" / "finetune.csv",
    "test_size": 0.2,
    "base_model": ROOT / "models" / "visobert_vsmec_vsfc",   # model đã có đầu phân loại 3 nhãn
    "adapter_dir": ROOT / "models" / "visobert_uit_lora",
    "merged_dir": ROOT / "models" / "visobert_uit",
    "max_length": 256,          # post confession dài hơn câu VSMEC/VSFC

    # ---- LoRA
    "lora_r": 8,                # hạng ma trận LoRA; tăng 16 nếu underfit
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    # Các lớp Linear trong encoder được gắn LoRA (regex, để không đụng vào classifier.dense).
    # Bản nhẹ hơn: r".*encoder.*\.(query|value)"
    "target_modules": r".*encoder\.layer\.\d+\.(attention\.self\.(query|key|value)|attention\.output\.dense"
                      r"|intermediate\.dense|output\.dense)",
    "train_classifier": True,   # train đầy đủ lớp phân loại (modules_to_save)

    # ---- Huấn luyện
    "epochs": 15,
    "batch_size": 16,
    "lr": 2e-4,                 # LoRA thường cần lr cao hơn full fine-tune (1e-4 .. 5e-4)
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "max_grad_norm": 1.0,
    "class_weight": True,
    "label_smoothing": 0.1,
    "fp16": True,
    "seed": 42,
}
# =============================================================================


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(texts, labels, tok, cfg, shuffle):
    def collate(batch):
        xs, ys = zip(*batch)
        enc = tok(list(xs), max_length=cfg["max_length"])
        enc["labels"] = torch.tensor(ys)
        return enc
    return DataLoader(list(zip(texts, labels)), batch_size=cfg["batch_size"], shuffle=shuffle, collate_fn=collate)


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


def compute_metrics(y_true, probs, losses, labels):
    y_pred = probs.argmax(1)
    m = {"n": len(y_true), "accuracy": accuracy_score(y_true, y_pred)}
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average=avg, zero_division=0)
        m.update({f"{avg}_precision": p, f"{avg}_recall": r, f"{avg}_f1": f})
    m["mcc"] = matthews_corrcoef(y_true, y_pred)
    m["cohen_kappa"] = cohen_kappa_score(y_true, y_pred)
    try:
        m["macro_roc_auc_ovr"] = roc_auc_score(y_true, probs, multi_class="ovr", average="macro",
                                               labels=range(len(labels)))
    except ValueError:
        pass
    _, _, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=range(len(labels)), zero_division=0)
    m.update({f"f1_{l}": f[i] for i, l in enumerate(labels)})
    m["loss"] = losses.mean()
    return {k: (v if k == "n" else round(float(v), 4)) for k, v in m.items()}


def print_report(title, y_true, probs, losses, labels):
    m = compute_metrics(y_true, probs, losses, labels)
    y_pred = probs.argmax(1)
    print(f"\n{'=' * 70}\n{title} ({len(y_true)} câu)\n{'=' * 70}")
    print(f"Accuracy          : {m['accuracy']:.4f}")
    print(f"Macro    P/R/F1   : {m['macro_precision']:.4f} / {m['macro_recall']:.4f} / {m['macro_f1']:.4f}")
    print(f"Weighted P/R/F1   : {m['weighted_precision']:.4f} / {m['weighted_recall']:.4f} / {m['weighted_f1']:.4f}")
    print(f"MCC               : {m['mcc']:.4f}")
    print(f"Cohen's kappa     : {m['cohen_kappa']:.4f}")
    if "macro_roc_auc_ovr" in m:
        print(f"Macro ROC-AUC(OvR): {m['macro_roc_auc_ovr']:.4f}")
    print(f"Loss              : {m['loss']:.4f}\n")
    print(classification_report(y_true, y_pred, labels=range(len(labels)), target_names=labels,
                                digits=4, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    print("Confusion matrix (hàng = thật, cột = dự đoán):")
    print(pd.DataFrame(cm, index=[f"thật_{l}" for l in labels], columns=[f"đoán_{l}" for l in labels]).to_string())
    return m


def main(cfg=CONFIG):
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = cfg["fp16"] and device == "cuda"
    print(f"Thiết bị: {device}" + (" (fp16)" if use_fp16 else ""))

    # ---------------------------------------------------------------- Dữ liệu
    tok = ViSoBERTTokenizer.from_pretrained(cfg["base_model"])
    base = AutoModelForSequenceClassification.from_pretrained(cfg["base_model"]).to(device)
    labels = [base.config.id2label[i] for i in range(base.config.num_labels)]
    label2id = {l: i for i, l in enumerate(labels)}

    df = pd.read_csv(cfg["data_path"], encoding="utf-8-sig")
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["label"].isin(labels) & df["text"].notna()].reset_index(drop=True)
    train_df, test_df = train_test_split(df, test_size=cfg["test_size"], stratify=df["label"],
                                         random_state=cfg["seed"])
    print("\nSố câu train / test theo nhãn:")
    print(pd.DataFrame({"train": train_df["label"].value_counts(), "test": test_df["label"].value_counts()})
          .reindex(labels).fillna(0).astype(int).to_string())

    tr_x, tr_y = train_df["text"].tolist(), train_df["label"].map(label2id).tolist()
    te_x, te_y = test_df["text"].tolist(), test_df["label"].map(label2id).to_numpy()
    train_dl = make_loader(tr_x, tr_y, tok, cfg, shuffle=True)
    test_dl = make_loader(te_x, te_y.tolist(), tok, cfg, shuffle=False)

    # ---------------------------------------------------------------- Điểm mốc trước fine-tune
    before_probs, before_losses = predict_proba(base, test_dl, device, use_fp16)
    before = print_report("TEST - TRƯỚC fine-tune (model gốc)", te_y, before_probs, before_losses, labels)

    # ---------------------------------------------------------------- Gắn LoRA
    lora_cfg = LoraConfig(task_type="SEQ_CLS", r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"],
                          lora_dropout=cfg["lora_dropout"], target_modules=cfg["target_modules"],
                          modules_to_save=["classifier"] if cfg["train_classifier"] else None)
    model = get_peft_model(base, lora_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\nLoRA: train {trainable:,} / {total:,} tham số ({trainable / total:.2%})")

    weight = None
    if cfg["class_weight"]:
        counts = np.bincount(tr_y, minlength=len(labels)).clip(min=1)
        weight = torch.tensor(len(tr_y) / (len(labels) * counts), dtype=torch.float, device=device)
        print("Trọng số lớp:", {l: round(w, 2) for l, w in zip(labels, weight.tolist())})
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight, label_smoothing=cfg["label_smoothing"])

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps = len(train_dl) * cfg["epochs"]
    sched = get_linear_schedule_with_warmup(opt, int(cfg["warmup_ratio"] * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    # ---------------------------------------------------------------- Train (số epoch cố định)
    print(f"\n{'epoch':>5} {'train_loss':>10} {'train_acc':>9} {'time':>6}")
    history = []
    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        total_loss, correct, n, t0 = 0.0, 0, 0, time.time()
        for b in train_dl:
            b = {k: v.to(device) for k, v in b.items()}
            y = b.pop("labels")
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_fp16):
                logits = model(**b).logits.float()
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
            scaler.step(opt)
            scaler.update()
            sched.step()
            opt.zero_grad()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(-1) == y).sum().item()
            n += len(y)
        history.append({"epoch": ep, "train_loss": round(total_loss / n, 4), "train_acc": round(correct / n, 4)})
        print(f"{ep:>5} {total_loss / n:>10.4f} {correct / n:>9.4f} {time.time() - t0:>5.1f}s")

    # ---------------------------------------------------------------- Đánh giá + lưu
    after_probs, after_losses = predict_proba(model, test_dl, device, use_fp16)
    after = print_report("TEST - SAU fine-tune LoRA", te_y, after_probs, after_losses, labels)

    print(f"\n{'=' * 70}\nSO SÁNH TRƯỚC / SAU trên cùng {len(te_y)} câu test\n{'=' * 70}")
    keys = ["accuracy", "macro_f1", "weighted_f1", "mcc", "f1_negative", "f1_neutral", "f1_positive"]
    cmp = pd.DataFrame({"trước": {k: before.get(k) for k in keys}, "sau": {k: after.get(k) for k in keys}})
    cmp["chênh"] = (cmp["sau"] - cmp["trước"]).round(4)
    print(cmp.to_string())

    after_pred = after_probs.argmax(1)
    changed = test_df.assign(trước=[labels[i] for i in before_probs.argmax(1)],
                             sau=[labels[i] for i in after_pred])
    changed = changed[changed["trước"] != changed["sau"]]
    print(f"\nCâu test đổi dự đoán sau fine-tune ({len(changed)}):")
    for _, r in changed.iterrows():
        mark = "✔" if r["sau"] == r["label"] else "✘"
        print(f"  {mark} thật={r['label']:<8} {r['trước']:>8} -> {r['sau']:<8} | {r['text'][:90]}")

    adapter_dir, merged_dir = Path(cfg["adapter_dir"]), Path(cfg["merged_dir"])
    model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    merged = model.merge_and_unload()
    merged.save_pretrained(merged_dir)
    tok.save_pretrained(merged_dir)

    pd.concat([train_df.assign(split="train"), test_df.assign(split="test")]).to_csv(
        merged_dir / "split.csv", index=False, encoding="utf-8-sig")
    test_df.assign(pred=[labels[i] for i in after_pred], conf=after_probs.max(1).round(3),
                   pred_before=[labels[i] for i in before_probs.argmax(1)]).to_csv(
        merged_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    meta = {"config": {k: str(v) if isinstance(v, Path) else v for k, v in cfg.items()},
            "history": history, "test_before": before, "test_after": after}
    (merged_dir / "metrics.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nĐã lưu adapter LoRA -> {adapter_dir}")
    print(f"Đã lưu model gộp    -> {merged_dir} (dùng cho predict.py)")


if __name__ == "__main__":
    main()
