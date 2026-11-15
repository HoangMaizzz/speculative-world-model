# Đo tái sử dụng nhánh reject của EAGLE‑3

Entrypoint `scripts/benchmark_eagle3_reuse.py`. Mặc định lấy 100 câu từ MATH500 test và 100 câu từ GSM8K main test, seed 42, cùng sampler và chat template của benchmark AR‑AR. Ngân sách sinh là 256 token mỗi câu. Một cell Kaggle tải code, hai checkpoint công khai và dataset. Chọn GPU T4 x2 và bật Internet. Có thể thay `--count 100` bằng `--count 5` và dùng thư mục output riêng để chạy pilot 10 câu.

## Runtime

- Dùng model Qwen2.5‑7B‑Instruct đã khóa trong `source-lock.json` và head Thoughtworks tại `ff17dda64a036cf5bd7bc56c0ab728325f1c0d0b`. Code tạo cây, chọn đường greedy và cập nhật KV lấy từ SafeAILab/EAGLE `cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b`.
- Adapter lấy đúng ba hidden state sau layer có chỉ số 0-based 1, 13, 24 như mã huấn luyện SpecJAX. Head EAGLE‑3 một block dùng embedding Qwen. Nạp đủ trọng số và kiểm tra offset `d2t` khớp tập target ID trong `t2d` trước khi chạy.
- FP16, greedy, batch 1. GPU0 chứa embedding và 18 layer đầu của verifier. GPU1 chứa 10 layer cuối, norm, LM head và nguyên drafter. Verifier dùng eager attention để áp dụng tree mask. Giữ KV của đường đã chọn trong cả verifier lẫn drafter.
- Mỗi cây có 32 node tính cả root do verifier sinh, `top_k=4`, `depth=7` theo tham số của repo EAGLE. Báo cáo độ sâu thực tế trong `summary.json`. Một nhánh chứa tối đa 8 token drafter vì lượt mở rộng đầu tiên đi trước 7 bước `depth`. Nhật ký đếm root riêng với số token nháp được accept.

## Đo khả năng tái sử dụng

1. Chạy một lượt EAGLE‑3 native và một lượt Qwen greedy từng token trên mỗi câu. Nhánh EAGLE dùng quyết định của verifier trong chính lượt chạy. Preflight dùng một câu ngắn và một câu từ mỗi tập, ghi đối chiếu token, kiểm tra cấu trúc cây và KV. Nếu cả ba đều lệch AR, lưu ZIP preflight để chẩn đoán và dừng.
2. Ghi mọi node được gửi sang verifier, danh sách cha–con và đường đã chọn. Trên từng nhánh, bỏ token sai đầu tiên và xét phần thật sự từng được drafter sinh tiếp trên nhánh đó, cắt trước EOS.
3. Sau token gốc của lượt kế tiếp do verifier cung cấp, so với ba vùng đích dài tối đa 10, 32 và 64 token của AR. Dùng những lượt mà prefix đến hết token gốc này vẫn khớp với AR. Lượt lệch, dừng ở EOS hoặc vùng đích dưới 4 token có bộ đếm riêng. Danh sách vòng và đầu ra native vẫn có trong ZIP.
4. Một anchor dài ít nhất 4 token phải khớp toàn bộ theo ID. Một anchor sai 1 token đóng góp 0. Các anchor đúng chồng nhau hoặc xuất hiện ở nhiều nhánh chỉ phủ mỗi vị trí đích một lần. Báo cáo các lượt tìm được anchor, độ phủ cùng vị trí, độ phủ sau shift, phần tăng thêm do shift, và phân bố chiều dài. Tính cùng mẫu số cho phạm vi nhánh được chọn và toàn bộ cây. Nhánh phụ có thể phân kỳ sớm hơn điểm sửa của nhánh được chọn. Tọa độ nguồn của nó có thể âm vì cả đích và nguồn đều lấy prefix sau token sửa làm mốc.

Đây là đo oracle trên trace. Thời gian trong báo cáo là thời gian của benchmark có thu trace. Đo được một anchor oracle giúp chọn thí nghiệm chèn anchor và sinh bridge tiếp theo.

## Kết quả được lưu

- `summary.json` có tổng hợp chung và theo tập, trạng thái chạy, manifest khóa checkpoint và cấu hình.
- `cases.jsonl` có đầu ra AR, cây nháp mỗi lượt, quyết định native và các đoạn khớp.
- `prompts.json`, `preflight.json` và `checkpoint.sqlite` dùng để kiểm tra và tiếp tục trong cùng thư mục làm việc. SQLite ghi mỗi câu, ZIP cập nhật sau mỗi 20 câu và khi kết thúc. Budget mặc định 6 giờ cho tiến trình benchmark.
