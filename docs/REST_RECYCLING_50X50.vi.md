# REST và Token Recycling trên 50 MATH‑500 + 50 HumanEval

Benchmark chạy decoder REST và Token Recycling từ hai commit chính chủ, cùng greedy AR của Hugging Face. Cả ba dùng **Vicuna‑7B‑v1.3 FP16, batch 1** và nhận đúng cùng prompt token IDs. Với hai GPU CUDA, mỗi GPU giữ 16 lớp và KV cache của các lớp đó. Thời gian đo gồm cả truyền dữ liệu giữa hai GPU. Với một GPU, toàn bộ model và cache nằm trên GPU đó. Manifest ghi cách chia lớp, checkpoint và dữ liệu. [REST](https://github.com/FasterDecoding/REST/tree/154d0f0ae45de5d8a9cbd1c696f6ad5da10904bf) · [Token Recycling](https://github.com/Luowaterbi/TokenRecycling/tree/1b4c05cc642d111f12c61d4a99950d737121d9fe)

| | MATH‑500 | HumanEval |
|---|---|---|
| Số câu | 50 đầu theo thứ tự sample 100 của Long64, seed 42 | 50 task từ file HumanEval của REST, seed 42 |
| Prompt | Đề toán, yêu cầu trình bày lời giải và đặt đáp số trong `\boxed{}` | Prompt Python nguyên bản của HumanEval |
| Giới hạn sinh | 1.024 token | 512 token |

REST dùng ShareGPT nhỏ cho cả hai tập, với cùng tokenizer Vicuna. Cấu hình tác giả có suffix 16→2, tối đa 10 token continuation, và tối đa 64 nút draft. Kho được dựng một lần ở thư mục cache và giữ cố định khi đo. Script không đưa lời giải hay test case của hai benchmark vào index. Code không kiểm định trùng lặp của corpus ShareGPT với hai benchmark.

Token Recycling dùng cây `2.2.2`, 80 nút draft và một root, top‑8 GPU, cập nhật cả nhánh reject theo code tác giả. Mỗi arm chạy một prompt tự viết ngoài tập đo cho mỗi dataset với ngân sách 32 token. Bảng Recycling được xóa sau warmup và tích lũy qua 50 câu trong từng dataset. Giữ nguyên thứ tự câu cho cả ba arm.

## Cell Kaggle

Gắn file `run_rest_recycling_50x50.py` dưới dạng một Kaggle Dataset, bật Internet và chọn **GPU T4 ×2** cho phiên chạy. Nếu dùng **Save & Run All**, kiểm tra accelerator trong **Advanced Settings** của cửa sổ lưu. Có thể dùng một GPU với đủ VRAM, ví dụ GPU 24 GiB. Để `SMOKE = False` chạy đủ 100 câu, đổi thành `True` để kiểm tra khởi động trên 2 + 2 câu với giới hạn 64 token.

```python
from pathlib import Path
import subprocess
import sys

SMOKE = False
gpu = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv"],
    capture_output=True, text=True,
)
if gpu.returncode != 0:
    raise RuntimeError("Phiên Kaggle chưa truy cập được GPU. " + (gpu.stderr or gpu.stdout).strip())
print(gpu.stdout.strip(), flush=True)

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

Dòng `CUDA preflight passed` xác nhận chính PyTorch trong môi trường benchmark đã khởi tạo được GPU. Nếu `nvidia-smi` thấy GPU và worker vẫn dừng, xem dòng `CUDA preflight` cùng `CUDA error` để xác định lỗi CUDA hoặc cấu hình thiết bị.

KV cache REST và Token Recycling được cấp theo độ dài prompt lớn nhất cộng ngân sách sinh và phần dự phòng cho cây. Việc giữ đường đã accept dùng cùng chỉ số của decoder gốc trên từng khối KV. Nếu GPU hết bộ nhớ trong lúc đo, phiên ghi ZIP với các dòng đã đo và `execution_status.json` có trạng thái `partial_cuda_oom`. Tải ZIP đó để giữ dữ liệu. Phiên đo đầy đủ sinh `summary.json` với trạng thái `complete`.

## Đọc kết quả

- `summary.json` báo số output khớp token với AR, tokens/s, token commit mỗi decode forward, số token draft accept mỗi decode forward, và speedup theo dataset. Speedup toàn nhóm chỉ hiện khi mọi câu khớp AR. Có số riêng trên tập con khớp AR và danh sách vị trí khác đầu tiên.
- `ar.jsonl`, `rest.jsonl`, `recycling.jsonl` giữ text, token IDs, latency có đồng bộ GPU từ trước prefill đến khi decoder trả về, và số liệu gốc từng vòng. Thời gian dựng kho, tải model, warmup và tokenize nằm ngoài phép đo.
- Hai decoder gốc có thể sinh quá ngân sách trong vòng cuối. Output để so sánh dừng ở EOS đầu tiên hoặc giới hạn chung. Thời gian vẫn tính cả vòng cuối, và số token dư được ghi. Với Recycling, runner khôi phục EOS nếu code gốc đếm EOS mà không append nó. Runner cấp thêm một lượt lặp tối đa vì vòng đầu có thể chỉ xử lý token cuối prompt.
- Benchmark đo tốc độ và tính nhất quán với AR. HumanEval không tự chạy code sinh ra để chấm pass@1. MATH‑500 không chấm tương đương đáp số.
