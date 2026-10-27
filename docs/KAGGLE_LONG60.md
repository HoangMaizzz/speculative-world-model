# Kaggle: draft60, 100 MATH +100 GSM8K

Bật **GPU T4 x2** và **Internet ON**. Dán toàn bộ cell bên dưới rồi Run hoặc Save Version / Run All. Không cần input ZIP cũ hoặc GitHub token.

Runtime pin: `31c6596ab6f56ccd07e8e1d4779f23b25674bc1a`. Hai model FP16 như trước: Fast-dLLM-v2-1.5B trên GPU1; Qwen2.5-7B-Instruct chia 18/10 layers giữa GPU0/GPU1. Native draft60, một MASK mỗi unmask-forward, output tối đa256token/câu. Xem [thiết kế](LONG60_PROTOCOL.vi.md).

```python
# Kaggle Settings: GPU T4 x2, Internet ON. Chạy độc lập trong một cell.
import hashlib
import urllib.request

REVISION = "31c6596ab6f56ccd07e8e1d4779f23b25674bc1a"
EXPECTED_SHA256 = "82765b1f1d7294e516d61c963ffeee90e1f905ce32207f3b4c910d1d53edc12d"
COUNT_PER_DATASET = 100
MAX_NEW_TOKENS = 256
RESUME_ROOT = None  # Phiên mới: đường dẫn ZIP/folder của long60_checkpoint.

url = (
    "https://raw.githubusercontent.com/HoangMaizzz/speculative-world-model/"
    + REVISION + "/scripts/kaggle_long60_cell.py"
)
print("Loading pinned draft60 launcher:", REVISION, flush=True)
with urllib.request.urlopen(url, timeout=60) as response:
    source = response.read()
if hashlib.sha256(source).hexdigest() != EXPECTED_SHA256:
    raise RuntimeError("Launcher checksum mismatch")

scope = {
    "__name__": "__main__",
    "LONG60_REVISION": REVISION,
    "LONG60_COUNT": COUNT_PER_DATASET,
    "LONG60_MAX_NEW_TOKENS": MAX_NEW_TOKENS,
    "LONG60_RESUME": RESUME_ROOT,
}
exec(compile(source, url, "exec"), scope)
```

Output: `/kaggle/working/long60/long60_checkpoint.zip` (hoặc thư mục có revision suffix được in trong log). Cell in link tải ZIP khi hoàn tất, khi dừng theo budget hoặc khi còn checkpoint sau lỗi. ZIP có raw token IDs/confidences/traces, canonical audits, từng đoạn khớp và bridge, CSV/JSON/HTML.

Soft budget6giờ; mỗi round lưu SQLite, ZIP định kỳ và sau từng câu. Rerun cùng cell trong cùng session tiếp tục local checkpoint. Phiên mới: upload/unzip ZIP B60 rồi đặt `RESUME_ROOT` tới đường dẫn đó. Không dùng `suffix_checkpoint`/Test3/Test4 làm resume B60. Giữ nguyên revision/count/output cap; numerical identity của GPU/packages/tokenizer cũng được kiểm tra.

Cell tự chạy CPU checks, rồi model thật kiểm tra cached/full B60 và target consistency trước Main. Sau mỗi câu, target greedy độc lập kiểm tra output/acceptance; mismatch dừng và giữ evidence. Đây là bản thu thập dữ liệu về shifted spans, chưa triển khai retrieval và chưa dùng để đo serving speedup.

