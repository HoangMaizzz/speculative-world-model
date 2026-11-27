# Benchmark cây retrieval với phản hồi 7B

Lượt mặc định chạy 50 MATH-500 và 50 GSM8K theo đúng thứ tự dữ liệu cố định của benchmark Ouroboros trước, tối đa 256 token/câu. Nạp cùng Qwen2.5-1.5B và 7B FP16 trên hai T4. Greedy 7B chạy tuần tự làm chuẩn riêng cho từng câu. Dữ liệu đầu vào là datatest đã dùng trước, gồm retrieval_alignment_data.json và selectors.joblib, hoặc ZIP ban đầu chứa hai tệp này.

| Cấu hình | Quyết định |
|---|---|
| base | Retrieval thường với corpus, lịch sử và bốn suffix gần nhất |
| boost | Cho token 7B cũ tham gia xếp hạng node. Giữ đường greedy 1.5B |
| fork | Cùng cây với boost. Gửi thêm tối đa một nhánh 4 token, lấy phần nối từ logits 1.5B đã tính dưới đúng prefix của nhánh |
| switch | Cùng cây với boost. Cho phép đổi một lần sang node ứng viên đã có rồi tiếp tục dưới prefix mới |
| greedy7b | 7B sinh từng token, dùng cùng placement và cache |

Cả bốn cấu hình retrieval có draft dài tối đa 16 và tối đa 16 node trong mỗi packet 1.5B. Mức ưu tiên phản hồi 7B là 1.5 cho suffix mới nhất, giảm dần sau đó. Ngưỡng cũ của 7B là 0.5. Fork dùng ngưỡng chênh điểm 0.025, switch dùng 0.4. Cả hai lấy top-16 của Q và top-32 của P. Các ngưỡng cố định từ phép thử trước. Kết quả lưu cả đầu ra khớp với baseline và greedy tuần tự. Cấu hình boost bao gồm chi phí xuất logits và ghi lại phản hồi như hai phương án có hành động. Thứ tự arm đảo xác định theo ID câu. Lượt warm-up và tải checkpoint nằm ngoài thời gian mỗi câu.

Báo cáo thời gian đầu cuối trên toàn bộ câu và trên tập ghép cặp có cùng token đầu ra. Chỉ số exact_to_greedy7b giới hạn phép so tốc độ theo cặp vào các câu hai phương pháp đều khớp chuẩn greedy tuần tự. Sai khác được ghi theo vị trí token bắt đầu từ 0. draft_forwards là các lần dựng draft chính, q_fill_calls đếm riêng các lần bổ sung cache. Thời gian draft bao gồm cả bước đọc logits, chọn hành động và bù cache. Không có bước AR regrow dành riêng cho fork. Phép thử không chấm điểm đáp án toán và tập câu đã được xem trong benchmark trước.

Mỗi cặp câu/cấu hình được ghi thành checkpoint riêng. Chạy lại cùng commit, tham số, cấu hình GPU và dữ liệu để tiếp tục nếu thư mục /kaggle/working còn giữ nguyên. ZIP cập nhật sau câu đầu, mỗi 10 câu và khi hoàn tất. ZIP chứa manifest, summary, per_prompt.csv và dữ liệu token từng câu.
