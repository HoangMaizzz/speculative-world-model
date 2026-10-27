# Kaggle: draft64 → verify, 100 MATH +100 GSM8K

Bật **GPU T4 x2**, **Internet ON**, rồi chạy toàn bộ cell bên dưới. Đây là SD bình thường: mỗi vòng draft mới64tokens, verify, commit prefix đúng+correction/bonus, bỏ phần còn lại. **Không cố định anchor và không mở rộng suffix cũ.** Phân tích đoạn/bridge chỉ chạy offline sau generation và canonical audit.

Runtime `b1d0b34e41b23b0748d202a26a0105b65a236066`; fixed FP16 models như trước. Native active8/physical32; mỗi unmask forward điền đúng một MASK. Output tối đa256tokens/câu hoặc EOS. Xem [protocol](LONG64_PROTOCOL.vi.md).

```python
# Kaggle: GPU T4 x2, Internet ON. Cell độc lập; draft → verify bình thường.
import hashlib
import urllib.request

REVISION = "b1d0b34e41b23b0748d202a26a0105b65a236066"
EXPECTED_SHA256 = "373e6cab1120a1a503ea195cc5d3afad1da08d164b24b8ce1ac1bf4d0cd093d1"
COUNT_PER_DATASET = 100
MAX_NEW_TOKENS = 256
RESUME_ROOT = None  # Chỉ dùng ZIP/folder checkpoint B64 khi cần resume.

url = (
    "https://raw.githubusercontent.com/HoangMaizzz/speculative-world-model/"
    + REVISION + "/scripts/kaggle_long64_cell.py"
)
print("Loading pinned draft64 launcher:", REVISION, flush=True)
with urllib.request.urlopen(url, timeout=60) as response:
    source = response.read()
if hashlib.sha256(source).hexdigest() != EXPECTED_SHA256:
    raise RuntimeError("Launcher checksum mismatch")

scope = {
    "__name__": "__main__",
    "LONG64_REVISION": REVISION,
    "LONG64_COUNT": COUNT_PER_DATASET,
    "LONG64_MAX_NEW_TOKENS": MAX_NEW_TOKENS,
    "LONG64_RESUME": RESUME_ROOT,
}
exec(compile(source, url, "exec"), scope)
```

ZIP mặc định: `/kaggle/working/long64/long64_checkpoint.zip`. Có log, heartbeat, SQLite mỗi round, ZIP định kỳ và sau mỗi prompt; soft budget6giờ có thể resume. Session mới upload ZIPB64, đặt `RESUME_ROOT` tới ZIP/folder đã unzip; không dùng ZIPB60/Test4.

Không cần input ZIP cũ để bắt đầu hoặc GitHub token. Preflight kiểm model thật trước Main; GPU results B64 chưa có ở lúc phát hành. Phân tích offline dùng horizon63, shifts±8/16/32/63; các runs và bridges có rawIDs/CSV/JSON/HTML.

