# Redraft 64 ô từ hậu tố AR–AR bằng một forward

Đầu vào là ZIP kết quả AR–AR64 của lượt chạy 100 MATH500 + 100 GSM8K tại `cee297a`. Từ mỗi câu lấy ca reject đủ điều kiện đầu tiên sau khi correction đã khớp Qwen 7B tuần tự. Có 199 ca gồm 99 MATH500 và 100 GSM8K. MATH500/338 không có reject. Không chạy lại rollout AR–AR.

Cùng corrected prefix cho ba nhánh. Fresh để 64 ô mask. Oracle chép chuỗi anchor nonzero-shift dài ít nhất 4 token tương thích có tổng số token lớn nhất. Nhánh predicted chép anchor ở đúng vị trí do head chọn. Anchor sai của model vẫn giữ nguyên. Fast-dLLM v2 1.5B xử lý 64 ô trong một forward với attention toàn canvas cho các hàng dự đoán, Qwen2.5-7B-Instruct verify bằng một forward trên KV của corrected prefix. Stop tại EOS/budget lưu trong ZIP. Thứ tự ba nhánh được xáo trộn cố định theo câu.

## Head đặt anchor

HGB Classifier dùng các feature vị trí và thống kê logits của Qwen 7B trên draft AR cũ. Replay một lần trên mỗi ca để lấy feature. Candidate gồm đoạn nguồn dài 4–8 token và vị trí đích khác vị trí nguồn. Nhãn đúng khi mọi token khớp ở vị trí đích. Train dùng toàn bộ positive và tối đa 512 negative ngẫu nhiên mỗi ca.

Năm fold theo prompt, phân tầng dataset. Mỗi lần dùng ba fold để train, một fold để chọn threshold, một fold độc lập để đánh giá. Threshold chọn theo số token đúng trừ hai lần số token sai trên validation. Có thể không giữ đoạn nào. Khi giữ nhiều đoạn, chúng không chồng lên nhau và tôn trọng thứ tự nguồn/đích. Qwen 7B gold của fold test chỉ dùng cho oracle và chấm kết quả.

## Đầu ra

ZIP lưu số token accept đối chiếu gold và accept native, thắng/hòa/thua so fresh, precision theo token và theo nguyên đoạn, coverage từ các đoạn khớp trọn, số lượt verify đủ 64, timing cho draft/verify, tạo KV, chọn anchor và replay feature. Timing theo nhánh là thời gian một lượt sửa prefix đã lưu. Replay feature là chi phí riêng của phép đo offline.

Kaggle cần T4 x2, Internet và `ar_ar_reuse_results.zip` gốc, hoặc thư mục đã giải nén giữ nguyên `cases.jsonl`, `prompts.json`, `summary.json`. Kết quả được checkpoint mỗi câu và xuất ZIP sau mỗi 25 ca, cũng như khi kết thúc.
