# Đo tái sử dụng hậu tố của Qwen AR–AR

Mặc định dùng Qwen2.5-1.5B-Instruct làm drafter và Qwen2.5-7B-Instruct làm verifier, greedy FP16, batch 1, 100 MATH500 + 100 GSM8K, seed 42. Dùng cùng bộ chọn câu và chat template như benchmark dLLM-AR32. Mỗi câu có ngân sách sinh 256 token. Mỗi lượt đề xuất tối đa 32 token, dừng khi gặp EOS hoặc hết ngân sách. Verifier kiểm tra toàn bộ đề xuất bằng một forward causal. Drafter và verifier giữ KV của prefix đã chốt và cắt KV sau điểm reject. Drafter chạy trên GPU1, verifier chia 18/10 tầng qua hai GPU.

Mỗi câu chạy thêm Qwen 7B greedy từng token để lấy tham chiếu độc lập. Thu native AR–AR trước khi đối chiếu oracle. Lưu proposal, điểm reject, token sửa hoặc bonus, token chốt và timing từng vòng. Lưu cả các câu có khác biệt số học giữa verifier batch và AR tuần tự. Khi đo tái sử dụng, chỉ dùng các vòng ở đó prefix **sau token sửa** vẫn trùng với gold từ AR.

## Tính hậu tố tái sử dụng

- Bỏ token sai đầu tiên. Nguồn là các token thực sự đã được submit ở bên phải điểm reject, cắt trước EOS. Đích bắt đầu ngay sau token sửa, giới hạn bởi cửa sổ **10** hoặc **32** token tiếp theo và cắt trước EOS.
- So khớp theo ID, lưu mọi đoạn liên tiếp cực đại dài ít nhất **4** token. Báo cáo giữ nguyên vị trí, khác vị trí, có một trong hai, và vị trí đích chỉ thu được thêm nhờ dịch. Dấu dương là dịch về phía sau trong chuỗi.
- Tính độ phủ bằng hợp các vị trí thuộc đoạn đúng trọn vẹn. Lưu cả độ phủ gộp trong từng cửa sổ và số vị trí duy nhất trên mỗi câu. Báo cáo riêng độ dài, độ dịch trong ±1/4/8/16, đoạn có nhiều hơn một ID, và các ca gần đầu cần bridge 0–3 token.
- Cây một nhánh ở lượt tiếp theo cho biết đích có nằm trong độ dài proposal, proposal có tự đoán lại, và phần chốt native có đi qua cả đoạn hay không. Đếm cả chấp nhận từ draft và chốt sau khi tính correction/bonus. Các ca gần đầu dùng tiêu chí đoạn chưa chốt trọn và có nhiều hơn một ID.
- Những lượt không reject, correction EOS, prefix lệch AR, hậu tố dưới 4 token, và tham chiếu dưới 4 token có bộ đếm riêng. Log giữ cả các cửa sổ đủ điều kiện mà không có đoạn khớp.

Đây là phép đo cơ hội oracle trên trace AR–AR. Kết quả timing là tốc độ baseline AR–AR, chưa đo tốc độ sau chèn anchor hay sinh bridge.

## Chạy và xem kết quả

Entrypoint là `scripts/benchmark_ar_ar_reuse.py`. Kaggle dùng Internet và GPU T4 x2 để tải hai checkpoint cùng hai dataset công khai. Mặc định chạy full. Có thể giảm `--count` để smoke, đổi thư mục output khi đổi tham số. Script ghi SQLite từng câu, xuất ZIP mỗi 20 câu và khi kết thúc, tiếp tục từ SQLite nếu chạy lại với cấu hình tương ứng.

- `summary.json` chứa kết quả theo tập và cả hai cửa sổ.
- `cases.jsonl` chứa đầu ra AR, vòng draft–verify và toàn bộ span.
- `prompts.json`, `preflight.json`, `checkpoint.sqlite` dùng để đối chiếu hoặc tiếp tục chạy.
