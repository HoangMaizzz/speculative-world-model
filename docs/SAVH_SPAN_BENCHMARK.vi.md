# So sánh cửa sổ trượt với head chấm cặp đầu–cuối

Bài test dùng ZIP kết quả `savh_ca813c7c0101_full.zip`. Đây là ZIP nhỏ chứa `features.npz`, `events.json` và labels từ SAVH pilot. SHA-256 là `5bcaf16e3a202a9eb05b0995559bde575d4c9fe5fd3ccab02314a110cfb83fc5`. Runner xác minh SHA-256 và tái tính nhãn exact từ token ID, đồng thời đối chiếu nhãn L4/L8 đã lưu. Input Long64 và ZIP verifier raw phục vụ các bước thu feature trước đó.

## Thiết kế thí nghiệm

Giữ split 164/19/17 prompt, tương ứng 3.012/344/350 event đủ điều kiện train/validation/test. Cohort yêu cầu source và reference dài ít nhất 8 token để cùng so sánh L4–L8. Nhãn dương yêu cầu nguyên đoạn xuất hiện liên tiếp ở một vị trí khác vị trí nguồn trong reference sau correction. Reference chỉ dùng cho nhãn và kiểm tra. Mỗi head có một bộ trọng số dùng chung cho mọi phương pháp sinh ứng viên trong cùng seed.

| Phương pháp | Cách sinh và chấm ứng viên |
|---|---|
| `fixed4` | Quét mọi cửa sổ 4 token, stride 1, rồi chọn nhiều đoạn. |
| `append` | Chấm cửa sổ 4 token, chọn 12 seed. Ưu tiên các vị trí bắt đầu cách nhau ít nhất 3 token rồi điền từ danh sách đã hoãn. Mở rộng cả trái và phải đến tổng độ dài 8. Loại trùng và chấm lại các đoạn mới trong một batch. |
| `dense` | Chấm mọi cặp đầu–cuối có độ dài 4–8 trong một batch. |

Các nhánh đều chọn đoạn không overlap ở nguồn theo thứ hạng confidence và tổng tối đa 16 token. Cả ba nhánh dùng cùng giới hạn độ dài, feature và trọng số trong phép so `append` với `dense`. `fixed4` là baseline bổ sung có tập độ dài cố định. Có thể chạy lại với `--max-length 12` hoặc `16` để đo ảnh hưởng của chiều dài.

| Head | Feature và kiến trúc |
|---|---|
| HGB | 15 scalar, thống kê mean/min/max và hai đầu đoạn. Chạy CPU. |
| MLP scalar | 15 scalar, chiếu mỗi token xuống 32 chiều, ghép hai đầu với mean pooling và embedding độ dài. |
| MLP hidden | Cùng MLP, thêm norm và PCA32 layer 27. |
| Conv1D hidden | Cùng feature và scorer với MLP hidden, thêm một Conv1D kernel 3. |
| Biaffine hidden | Cùng feature, tích hai projection đầu–cuối hạng 16, cộng score từ pooling và độ dài. |
| Attention hidden | Cùng feature và scorer với MLP hidden, thêm một lớp attention 4 head. |

Full fit một HGB và năm head PyTorch với ba seed, tổng **16 fits và 48 cặp head/phương pháp**. Chuẩn hóa fit trên train. PyTorch chọn epoch theo macro AP trên validation. HGB dùng CPU và các head PyTorch dùng thiết bị đã chọn, do đó phép so latency chính là `append` với `dense` trong từng head.

## Ngưỡng giữ với độ dài thay đổi và nhiều đoạn

1. Chia validation theo prompt thành nhóm hiệu chuẩn và nhóm chọn policy. Với mỗi head và mỗi route, fit `sigmoid(a * logit + bias[length])`, với `a >= 0`, trên đúng các ứng viên route đó sinh ra. Append tạo ứng viên bằng điểm seed 4 token trước bước này. Mỗi đoạn sau append được chấm nguyên đoạn và nhận bias của đúng độ dài.
2. Chạy hai họ policy. `common` dùng một ngưỡng chung trên xác suất đã hiệu chuẩn. `length_aware` so các profile đặt trước cho nhóm **4 / 5–6 / 7+ token**. Các profile dùng độ lệch `0/0/0`, `0/+.03/+.06`, `0/+.05/+.10` hoặc `0/-.03/-.06`. Với độ lệch `d` và ngưỡng cơ sở `t`, đoạn đạt nếu `p-d >= t`. Dùng cùng `p-d` để xếp hạng. Ví dụ, `t=.72` và profile `0/+.03/+.06` cho ba ngưỡng `.72/.75/.78`.
3. Duyệt theo thứ hạng, giữ đoạn khi không đè lên đoạn đã chọn và còn chỗ trong ngân sách 16 token. Một suffix có thể giữ 0 đến 4 đoạn. Một seed ngắn đủ ngưỡng vẫn có thể được giữ khi đoạn mở rộng không đạt ngưỡng. Khi hai đoạn overlap, selector giữ theo thứ hạng. Ngân sách là mức tối đa, có thể giữ ít hơn 16 token.
4. Trên nhóm validation dùng để chọn policy, tối đa hóa số token đúng với **cả precision theo đoạn và precision theo token** đạt mục tiêu 70/75/80/85/90%, tối thiểu 40 đoạn từ 6 prompt ở full. Trong họ `length_aware`, profile chung là một lựa chọn và được ưu tiên khi bằng số token đúng. Khóa profile và ngưỡng rồi áp nguyên sang test. Nếu thiếu điểm đủ điều kiện, ghi `no_validation_gate`. Khi đã có ngưỡng nhưng không giữ đoạn nào trên test, ghi `no_test_selections`.

## Các phép đo

1. Đo số ứng viên chấm và tỷ lệ vị trí token oracle có ít nhất một đoạn đúng trong tập ứng viên. Đây là trần trước bước chọn và ngưỡng.
2. Báo precision theo đoạn, precision theo token, số token đúng, recall trên union token oracle, tỷ lệ event có dự đoán và AP theo độ dài trên tập ứng viên. Một đoạn sai đóng góp 0 token đúng. AP ứng viên là số đo có điều kiện trên tập đã sinh.
3. Với mỗi ngưỡng đã khóa, báo số suffix giữ 0/1/2/3/4 đoạn và precision theo từng độ dài. Đường cong test của profile chung dùng để đối chiếu mô tả ở cùng precision thực đạt. Báo CI 95% ghép cặp theo prompt của chênh lệch token đúng/event giữa hai route tại từng mốc khóa, cùng với precision thực đạt của từng route.
4. Benchmark batch 1 trên tối đa 96 suffix test, warmup rồi lặp 3 lần và đảo thứ tự route giữa các lượt. Đo p50/p95 cả khi dùng policy `length_aware` đã khóa tại mốc 70% và khi chạy không lọc điểm. Nếu mốc 70% không có ngưỡng, báo fallback. Đồng bộ CUDA ở biên đo. Một lượt riêng tách thời gian encode, sinh ứng viên, chấm và chọn. Chi phí chuyển feature cache sang GPU và đưa điểm về CPU nằm trong phép đo neural head. Chi phí chạy Qwen/Fast-dLLM, thu hidden/raw logits, đặt anchor, repair và verify lại thuộc benchmark hệ thống riêng.

ZIP SAVH đã nén chỉ có scalar và PCA32 layer 27. `M_off`, hidden trước khi đọc token và fusion nhiều layer cần thêm raw feature archive. Bài test này dùng dữ liệu toán đã được xem để thăm dò, nên kết quả cần xác nhận lại trên prompt mới.

## Kaggle một cell

Gắn ZIP kết quả SAVH bằng Add Input. Bật Internet và chọn GPU T4 x2. Runner dùng một GPU cho head PyTorch, HGB chạy CPU. Có thể đặt `DEVICE = "cpu"` để chạy không cần GPU. Cell chạy smoke rồi full và đóng gói `summary.json` cùng `comparison.csv`.

```python
from pathlib import Path
import hashlib
import shutil
import subprocess
import sys
import urllib.request

INPUT = Path("/kaggle/input")  # hoặc đường dẫn trực tiếp đến ZIP SAVH full
DEVICE = "cuda"
MAX_LENGTH = 8
REV = "4545f6449c391c0044efb6cacd7a874a3c9a167b"
SHA = "598ef366791e42fd2079ba7665933d354d37e3566361c55f333acc6003d5b62e"

matches = [INPUT] if INPUT.is_file() else sorted(INPUT.rglob("savh_ca813c7c0101_full.zip"))
if len(matches) != 1:
    raise ValueError(f"Đặt INPUT thành đường dẫn trực tiếp đến ZIP kết quả SAVH full. Tìm thấy {matches}")

work = Path("/kaggle/working")
script = work / "benchmark_savh_spans.py"
url = f"https://raw.githubusercontent.com/HoangMaizzz/speculative-world-model/{REV}/scripts/{script.name}"
with urllib.request.urlopen(url, timeout=60) as response:
    payload = response.read()
if hashlib.sha256(payload).hexdigest() != SHA:
    raise RuntimeError("SHA-256 của runner không khớp commit đã pin")
script.write_bytes(payload)

base = [sys.executable, "-u", str(script), "--input", str(matches[0]),
        "--device", DEVICE, "--max-length", str(MAX_LENGTH)]
smoke = work / "savh_span_threshold_smoke"
full = work / "savh_span_threshold_compare"
subprocess.run(base + ["--output", str(smoke), "--smoke", "--latency-events", "8", "--repeats", "1"], check=True)
subprocess.run(base + ["--output", str(full)], check=True)
print("Kết quả:", shutil.make_archive(str(full), "zip", full))
```

`summary.json` có `complete: true` khi tất cả fit kết thúc. `comparison.csv` liệt kê các mốc validation và kết quả test từng head, từng seed, từng route. Output được cập nhật sau mỗi fit. Mỗi lần chạy thực hiện lại toàn bộ cấu hình được chọn. Các phép thử Python nằm ở `tests/test_savh_span_benchmark.py`.
