# Kaggle B64: bản sửa numerical verification

Runtime pin: `39890af682c068a51f5788594195f86e79a28575`. **100 MATH +100 GSM8K**, FP16 T4x2, draft64 mới mỗi vòng, không fixedanchor hoặc extend. Bật InternetON. Không cần inputZIP cũ.

Bản trước false-reject ở MATH327: drafter/canonical chọn ` a`, parallel target trả ` the` với top1–top2 margin.03125. Bản sửa giữ parallelverification và kiểm đường commit bằng actualcanonicalKV của cùngtarget trước khi emit. **Guard thêm serial target forwards**, nên đây là data-collection/correctness mode, không phải one-target-pass SD speed benchmark. Reference sau mỗi câu vẫn audit độc lập. Xem [protocol](LONG64_PROTOCOL.vi.md).

Chạy lại fresh bằng cell sau. Không đặt ZIPfailedb1d0b34e vàoRESUME_ROOT; output cũ được giữ nguyên.

```python
# Draft64 + canonical guard trước commit. Kaggle: T4 x2, Internet ON.
import hashlib
import urllib.request

REVISION = "39890af682c068a51f5788594195f86e79a28575"
EXPECTED_SHA256 = "0c0b14f198b6b4090268e691379fbdfbbbd6149ad5034534c89a8b86dafb36e9"
COUNT_PER_DATASET = 100
MAX_NEW_TOKENS = 256
RESUME_ROOT = None  # Bắt đầu mới; ZIP B64 cũ bị lỗi không tương thích mode này.

url = (
    "https://raw.githubusercontent.com/HoangMaizzz/speculative-world-model/"
    + REVISION + "/scripts/kaggle_long64_cell.py"
)
print("Loading guarded draft64 launcher:", REVISION, flush=True)
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

ZIP mới: `/kaggle/working/long64_guarded/long64_checkpoint.zip` (hoặc revision-suffixed folder được in trong log). Lưurawparallelfeedback,canonicalcheckedrows,extra forwards,numerical_disagreements.csv,allB64drafts/traces,canonicalaudit và shiftedspans/bridgesCSV/JSON/HTML.

SQLite mỗi round,ZIP180s+prompt/final,soft6h budget. Session mới chỉresumeZIP cùngrevision/config/mode. Mọi mismatch độc lập còn lại vẫn làm fail và giữevidence; không có gold substitution/INT8/offload. Local171tests pass; chưa có pretrainedT4run của fix tại lúc publish.
