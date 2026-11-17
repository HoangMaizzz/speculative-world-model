# Đo top K của verifier trên rejected suffix AR–AR

## Câu hỏi

Trong lượt verify **bản nháp cũ**, tại vị trí sau first rejection, token đúng của
continuation cuối cùng và token của rejected suffix cũ có nằm trong top K logits
hay không? K = 1, 2, 4, 8, 16, 32. Đây là logits teacher-forced theo nhánh
**chứa token bị reject**, không phải logits sau correction.

## Dữ liệu và cách phát lại

- Đầu vào: `ar_resync_e2e_results.zip` hoặc thư mục đã giải nén với
  `summary.json`, `cases.jsonl`, `prompts.json`. Dùng arm `fresh` gồm 100 MATH,
  100 GSM8K; không tạo bản nháp mới và không nạp drafter.
- Nạp đúng revision Qwen2.5-7B-Instruct từ manifest, FP16, split 18/10 trên
  Kaggle T4×2. Verifier KV chạy lại *mọi round* theo thứ tự để tái tạo prefix
  đúng và crop sau rejection. Chỉ phân tích các round có rejection.
- Mỗi round phải khớp trace cũ ở accepted length, first rejection, correction,
  bonus và committed tokens. Mismatch làm dừng, không âm thầm bỏ mẫu.
- Vị trí `i` so với proposal cũ có target tham chiếu là token output cuối cùng
  tại `offset+i` nếu vị trí đó tồn tại. Bên cạnh đó đo rank/xác suất token
  của suffix cũ ngay tại vị trí source cũ.
- Nhãn `source_reusable_anywhere4`: token cũ nằm trong ít nhất một run exact
  dài ≥4 xuất hiện tại *bất kỳ vị trí nào* trong 64 token tiếp theo của target.
  Nhãn này không có nghĩa là token cũ đúng tại cùng vị trí hoặc cả đường COPY
  sẽ được verifier accept.

## Kết quả lưu

`completed.jsonl` ghi top32 ID/xác suất, rank≤32 và xác suất của old/reference
token ở từng vị trí từ điểm reject đến hết proposal. Rank `null` nghĩa là
ngoài top32; reference `null` nghĩa là output đã kết thúc. Mỗi câu được ghi
ngay để có thể chạy tiếp khi Kaggle ngắt phiên. `summary.json` cho số lượng
top K theo dataset, khoảng cách sau rejection và nhãn source reusable.
`ar_suffix_topk_results.zip` đóng gói cả hai file.

## Diễn giải

Tỷ lệ top K cao của *reference token* sau reject chứng tỏ verifier vẫn gán
khối xác suất cho token đúng dù đọc nhánh cũ. Tỷ lệ top K cao của *old token*
trong nhóm `source_reusable_anywhere4` có thể làm feature để chọn source.
Không suy ra acceptance sau correction từ hai tỷ lệ này. Để kết luận khả năng
COPY ở vị trí mới, cần score proposal dưới corrected prefix và bridge hợp lệ.
Timing đo ở đây là chi phí thu thập tín hiệu, không phải tốc độ giải mã của
policy mới. Mã adapter được syntax/pure-data checked local; cần GPU Kaggle
để kiểm tra parity và báo tỷ lệ thực nghiệm.

