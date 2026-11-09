# Shift64 · kiểm tra shifted anchors trên 100 MATH + 100 GSM8K

Dùng đúng checkpoint B64 guarded hoàn tất (SHA256 `fb7e1ea13a0b6f6fce9f20d22c5270ab2c54e70cf83c59f7b95b31b44cacafca`). Không cần archive verifier features. Runner kiểm SHA trước khi load model. Tập câu hỏi trùng thí nghiệm B64, không phải test set mới.

Mỗi câu chọn rejection sớm nhất có correction thực sự được commit, còn rejected suffix không rỗng và có reference quan sát đủ 64 token hoặc tới EOS tự nhiên. Quy tắc không phụ thuộc việc tìm được anchor hay kết quả các nhánh. 200/200 câu đủ điều kiện. 88 MATH và 93 GSM8K có shifted anchor. Một GSM8K có reference kết thúc tự nhiên trước 64. Mọi nhánh vẫn sinh canvas 64, cắt proposal logic tại EOS.

| Nhánh | Anchor |
| --- | --- |
| Fresh64 | 64 MASK |
| Positional64 | Hard KEEP các token cũ đúng ngay vị trí gốc sau correction |
| Shifted64 | Hard KEEP các đoạn exact-match dài ít nhất 4, dịch khác 0 trong ±63 |

Shifted64 dùng chuỗi các maximal runs không chồng lấn, giữ thứ tự source và destination, tối đa tổng số token trong tập runs đó. Tie theo ít đoạn hơn, tổng độ dịch nhỏ hơn, rồi thứ tự vị trí. Không anchor EOS. Nếu không có shifted anchor, Shifted64 có cấu hình giống Fresh64. Positional64 là tham chiếu bổ sung với số anchor có thể khác. Phép so sánh chính là Shifted64 − Fresh64.

Cả ba nhánh bắt đầu ngay sau cùng correction. Đặt hard anchors một lần rồi sinh các vị trí MASK còn lại tới tổng 64, không nối thêm 64 token sau suffix. Native 32/8, strict một MASK mỗi unmask forward. Fast-dLLM-v2-1.5B và Qwen2.5-7B-Instruct FP16, T4 x2. Preflight B64 và kiểm cached/full với shifted anchors thực đầu tiên. Verifier chạy actual parallel score và canonical guard, khởi tạo lại đúng prefix từ prompt cho mỗi nhánh. Nếu kết quả lệch reference, giữ raw evidence và dừng.

Primary metric là accepted prefix, có EOS nếu được accept. Báo thêm content-only acceptance, win/tie/loss, paired bootstrap theo prompt 2.000 lần và số forwards. Summary tách MATH, GSM8K, pooled và nhóm có shifted anchor. Plan, raw draft, traces, anchors, verifier và kết quả giữ trong checkpoint. Đây là oracle intervention dùng target reference để chọn anchor, đo tiềm năng cải thiện draft, không đo policy online hoặc speedup.

Kaggle bật Internet và T4 x2, gắn `long64_checkpoint.zip` cũ. Launcher hỗ trợ smoke với LIMIT=2 (2 câu mỗi bộ), full với LIMIT=None. Runner lưu SQLite từng nhánh, tạo ZIP mỗi 10 câu và khi kết thúc hoặc checkpoint sau ngân sách mềm 6 giờ. Resume bằng cùng cấu hình, source ZIP và output ZIP nếu chuyển phiên Kaggle. Xem log và `execution_status.json` để phân biệt complete và budget_checkpoint.
