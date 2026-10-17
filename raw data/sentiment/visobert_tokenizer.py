"""Tokenizer cho uitnlp/visobert dùng trực tiếp SentencePiece.

Tokenizer của transformers 5.x chuyển đổi sai file sentencepiece.bpe.model của ViSoBERT:
"rất" bị cắt thành "▁r" + "ất" dù vocab có sẵn "▁rất". Class này tách từ bằng SentencePiece
gốc rồi đổi sang id theo quy ước XLM-R (fairseq): id = id_sentencepiece + 1,
<s>=0, <pad>=1, </s>=2, <unk>=3.
"""
import shutil
from pathlib import Path

import sentencepiece as spm
import torch

SPM_FILE = "sentencepiece.bpe.model"


class ViSoBERTTokenizer:
    bos_id, pad_id, eos_id, unk_id = 0, 1, 2, 3

    def __init__(self, spm_path):
        self.spm_path = str(spm_path)
        self.sp = spm.SentencePieceProcessor(model_file=self.spm_path)

    @classmethod
    def from_pretrained(cls, name_or_dir):
        local = Path(name_or_dir) / SPM_FILE
        if local.exists():
            return cls(local)
        from huggingface_hub import hf_hub_download
        return cls(hf_hub_download(str(name_or_dir), SPM_FILE))

    def save_pretrained(self, out_dir):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        shutil.copy(self.spm_path, Path(out_dir) / SPM_FILE)

    def tokenize(self, text):
        return self.sp.encode(text, out_type=str)

    def encode(self, text, max_length=256):
        ids = [i + 1 if i > 0 else self.unk_id for i in self.sp.encode(text)]
        return [self.bos_id] + ids[:max_length - 2] + [self.eos_id]

    def __call__(self, texts, max_length=256):
        encoded = [self.encode(t, max_length) for t in texts]
        width = max(map(len, encoded))
        input_ids = torch.full((len(encoded), width), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(encoded), width), dtype=torch.long)
        for i, ids in enumerate(encoded):
            input_ids[i, :len(ids)] = torch.tensor(ids)
            attention_mask[i, :len(ids)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}
