# REST và Token Recycling trên 50 MATH‑500 + 50 HumanEval

Benchmark chạy decoder REST và Token Recycling từ hai commit chính chủ, cùng greedy AR của Hugging Face. Cả ba dùng **Vicuna‑7B‑v1.3 FP16, batch 1, một GPU** và nhận đúng cùng prompt token IDs. Checkpoint và dữ liệu được ghi vào manifest. Đây là phép so trên checkpoint và hai tập đã chọn. [REST](https://github.com/FasterDecoding/REST/tree/154d0f0ae45de5d8a9cbd1c696f6ad5da10904bf) · [Token Recycling](https://github.com/Luowaterbi/TokenRecycling/tree/1b4c05cc642d111f12c61d4a99950d737121d9fe)

| | MATH‑500 | HumanEval |
|---|---|---|
| Số câu | 50 đầu theo thứ tự sample 100 của Long64, seed 42 | 50 task từ file HumanEval của REST, seed 42 |
| Prompt | Đề toán, yêu cầu trình bày lời giải và đặt đáp số trong `\boxed{}` | Prompt Python nguyên bản của HumanEval |
| Giới hạn sinh | 1.024 token | 512 token |

REST dùng ShareGPT nhỏ cho cả hai tập, với cùng tokenizer Vicuna. Cấu hình tác giả có suffix 16→2, tối đa 10 token continuation, và tối đa 64 nút draft. Kho được dựng một lần ở thư mục cache và giữ cố định khi đo. Script không đưa lời giải hay test case của hai benchmark vào index. Code không kiểm định trùng lặp của corpus ShareGPT với hai benchmark.

Token Recycling dùng cây `2.2.2`, 80 nút draft và một root, top‑8 GPU, cập nhật cả nhánh reject theo code tác giả. Mỗi arm chạy một prompt tự viết ngoài tập đo cho mỗi dataset với ngân sách 32 token. Bảng Recycling được xóa sau warmup và tích lũy qua 50 câu trong từng dataset. Giữ nguyên thứ tự câu cho cả ba arm.

## Cell Kaggle

Gắn file `run_rest_recycling_50x50.py` dưới dạng một Kaggle Dataset, bật Internet và GPU. Dùng **GPU 24 GiB trở lên** nếu có. Một T4 16 GB rất sát VRAM vì Vicuna FP16 và KV cache cấp phát sẵn. Launcher chỉ dùng một GPU. Để `SMOKE = False` chạy đủ 100 câu, đổi thành `True` để kiểm tra khởi động trên 2 + 2 câu với giới hạn 64 token.

```python
from pathlib import Path
import subprocess
import sys

SMOKE = False
scripts = sorted(Path("/kaggle/input").rglob("run_rest_recycling_50x50.py"))
if len(scripts) != 1:
    raise FileNotFoundError(f"Attach one run_rest_recycling_50x50.py. Found: {scripts}")

subprocess.check_call([sys.executable, "-m", "pip", "install", "uv==0.7.22"])
subprocess.check_call([
    sys.executable, "-u", str(scripts[0]),
    "--work", "/kaggle/temp/rest_recycling_50x50",
    "--output", "/kaggle/working/rest_recycling_50x50",
    *(["--smoke"] if SMOKE else []),
])
```

Model, ShareGPT và câu hỏi được tải tự động. Cần khoảng 13,5 GB cho trọng số, cộng môi trường CUDA và cache. Các lần chạy lại trong cùng phiên tái sử dụng model và index. Kết quả xuất thành một file ZIP trong `/kaggle/working/rest_recycling_50x50`.

## Đọc kết quả

- `summary.json` báo số output khớp token với AR, tokens/s, token commit mỗi decode forward, số token draft accept mỗi decode forward, và speedup theo dataset. Speedup toàn nhóm chỉ hiện khi mọi câu khớp AR. Có số riêng trên tập con khớp AR và danh sách vị trí khác đầu tiên.
- `ar.jsonl`, `rest.jsonl`, `recycling.jsonl` giữ text, token IDs, latency có đồng bộ GPU từ trước prefill đến khi decoder trả về, và số liệu gốc từng vòng. Thời gian dựng kho, tải model, warmup và tokenize nằm ngoài phép đo.
- Hai decoder gốc có thể sinh quá ngân sách trong vòng cuối. Output để so sánh dừng ở EOS đầu tiên hoặc giới hạn chung. Thời gian vẫn tính cả vòng cuối, và số token dư được ghi. Với Recycling, runner khôi phục EOS nếu code gốc đếm EOS mà không append nó. Runner cấp thêm một lượt lặp tối đa vì vòng đầu có thể chỉ xử lý token cuối prompt.
- Benchmark đo tốc độ và tính nhất quán với AR. HumanEval không tự chạy code sinh ra để chấm pass@1. MATH‑500 không chấm tương đương đáp số.
