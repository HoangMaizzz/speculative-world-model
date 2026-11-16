# Benchmark end-to-end tái sử dụng rejected suffix trong AR–AR

Qwen2.5-1.5B-Instruct và Qwen2.5-7B-Instruct, greedy FP16, batch 1, hai T4.
Mỗi câu có ngân sách output 256 token. Proposal dài tối đa 64 token, cắt tại
EOS hoặc giới hạn output. Verifier chạy cùng một cơ chế causal verify có KV
cache cho cả ba nhánh. Tất cả nhánh chạy trên từng câu, đảo thứ tự theo cả sáu
hoán vị. Thời gian theo câu gồm cả prefill, search, forward và cập nhật cache,
được đo sau khi model đã tải và warm up.

| Nhánh | Cơ chế |
|---|---|
| `fresh` | Qwen1.5B greedy tuần tự với KV cache |
| `checked_copy` | Tìm match ít nhất 4 token trong 16 token đầu của lượt mới. Thử một khối 4 token. Bỏ sớm nếu top-1 tiếp theo của drafter khác token COPY đầu tiên. Kiểm tra nhân quả cả khối, cắt KV khi lệch rồi tiếp tục tới ngân sách 64 |
| `direct_copy` | Chấm các đoạn 4/8/16 token từ vị trí đồng bộ trong 16 token đầu. Lấy ứng viên điểm cao nhất tại bước đầu vượt ngưỡng. Tính KV dưới prefix hiện tại và sinh tiếp, để Qwen7B duyệt COPY |

Nhánh direct dùng HGB hồi quy trên cận dưới của chênh lệch số token accept so
với full fresh. Head dùng 21 feature từ token và vị trí, không gọi thêm verifier.
Với một đoạn khác fresh và đúng tới cuối COPY, nhãn chỉ ghi phần accept quan
sát được tới cuối COPY. Khi toàn bộ đoạn giống fresh, nhãn delta bằng 0. Bộ train/
validation/test offline có 1.414/473/468 lượt tách theo câu. Ngưỡng 0,8 được
chọn trên validation theo cận dưới của gain, với tối thiểu 20 lượt kích hoạt để
có đủ lần thử cho nhánh thăm dò. Validation kích hoạt 21 lượt với cận dưới −5
token, test kích hoạt 26 lượt với cận dưới −34 token. Đây là cấu hình thăm dò, và
fresh vẫn là baseline được đo trên mọi câu.

Benchmark tải 100 câu MATH-500 và 100 câu GSM8K từ các revision dataset được
pin, seed 2718. Loại 200 ID đã dùng cho nghiên cứu offline và các câu có nội
dung giống chúng trong cùng dataset. Gắn checkpoint AR–AR64 hoàn chỉnh làm
input để launcher huấn luyện head trên CPU trong phiên Kaggle.

Đầu ra `ar_resync_e2e_results.zip` ghi báo cáo tổng hợp theo dataset và nhóm
gộp, chênh lệch paired, CI bootstrap theo câu, latency, token/s, số lượt verify,
accept/verify, số forward drafter, breakdown chi phí và parity đầu ra/proposal
so với fresh. Speedup chính là tổng thời gian fresh chia tổng thời gian nhánh
tương ứng, có thêm thống kê trên các câu đầu ra khớp. Log từng lượt chứa token
ID, điểm COPY, thời gian và token mục tiêu đã nhận. GPU preflight ghi parity
của một đoạn ép đi qua forward khối. Kiểm tra CPU chỉ xác nhận quy tắc cache,
fallback, EOS và checkpoint.

Launcher `scripts/kaggle_ar_resync_e2e.py` nhận `--revision`, `--trace`
(ZIP hoặc thư mục chứa `cases.jsonl` và `summary.json`), `--output`,
`--count` (mặc định 100 mỗi tập) và tùy chọn `--resume`. Hãy bật Internet và
chọn GPU T4 x2. Dùng `--count 5` cho pilot 10 câu. Benchmark lưu SQLite theo
từng câu hoàn chỉnh và cập nhật ZIP khoảng mỗi ba phút. Khi resume vào thư
mục mới, đưa ZIP kết quả hoặc thư mục đã giải nén chứa `checkpoint.sqlite`.
Giữ nguyên revision, số câu và cấu hình.
