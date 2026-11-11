# Soft Anchor Repair trên Kaggle

Pilot đo mức tăng token được Qwen chấp nhận khi mở một hoặc hai vị trí để Fast‑dLLM điền lại trong một anchor đã shift. Vị trí đặt được khóa offline. Drafter nhận prefix sau correction, các token cũ được giữ và các mask; tự điền toàn bộ bridge và mask trong một lượt tạo proposal 64 token. Qwen 7B xác minh thật và chạy canonical guard như Draft‑64 gốc.

Gắn đúng `soft_anchor_pilot_input.zip` làm Kaggle Input. Gói khoảng 16 KB chứa 48 ca đã khóa từ Draft‑64 và SAVH, gồm prefix token ID, anchor, thứ tự nghi ngờ từ head và nhãn audit. Cell nhận cả ZIP nguyên và `soft_anchor_plan.json` đã giải nén nếu đúng checksum. Model Fast‑dLLM v2 1.5B và Qwen2.5‑7B‑Instruct được tải từ các revision trong `source-lock.json`. Chọn T4 x2 và bật Internet.

```python
import hashlib
import urllib.request

REV = "d657bfdf72f54ec57999db6449f4b88c60e43bc5"
SHA = "8770b8be2c24877ad2048223dc1a8cc49851b8f03745489532d7326d59d8d375"
MODE = "pilot"  # "smoke" runs one case per cohort.

url = (f"https://raw.githubusercontent.com/HoangMaizzz/speculative-world-model/"
       f"{REV}/scripts/kaggle_soft_anchor_cell.py")
with urllib.request.urlopen(url, timeout=60) as response:
    source = response.read()
if hashlib.sha256(source).hexdigest() != SHA:
    raise RuntimeError("Launcher checksum mismatch")
exec(compile(source, url, "exec"), {
    "__name__": "__main__",
    "SOFT_ANCHOR_REVISION": REV,
    "SOFT_ANCHOR_MODE": MODE,
})
```

## Cohort và đối chứng

| Cohort | Số ca | Vai trò |
|---|---:|---|
| `one_error` | 24 | Mở một mask và so giữ cứng, ngẫu nhiên, điểm head, oracle mask, fresh64. |
| `exact` | 8 | Đo rủi ro sửa nhầm khi anchor vốn đúng ở đích. |
| `two_errors` | 4 | Thử mở hai mask trên đoạn dài ít nhất 6 token; có thêm đối chứng một mask. |
| `representative` | 12 | Mẫu đều theo dataset trước khi chọn anchor. Hai ca chỉ có fresh64 vì không có đoạn phù hợp. |

48 ca từ 16 prompt, tổng 222 lượt tạo proposal. MATH và GSM8K chia đều ở các cohort `one_error`, `exact` và `representative`. Cohort hai lỗi có ba GSM8K và một MATH do nguồn ứng viên ít. Tập test SAVH đã dùng thăm dò trước đây. Pilot cố định một anchor 4–8 token, bắt đầu sau bridge dài 1–7 token trong block vật lý 32 đầu tiên. Oracle chọn đích có Hamming nhỏ nhất trong phạm vi đó, hòa thì chọn đích sớm hơn. Nhánh oracle mask dùng vị trí lỗi. Toàn bộ token thay thế do drafter sinh. Nhánh chạy theo thứ tự xáo xác định từng ca. Cohort làm giàu và mẫu đều báo riêng.

## Đọc kết quả

Khi chạy xong, tải `soft_anchor_pilot_results.zip` từ thư mục được cell in ra, mặc định `/kaggle/working/soft_anchor_pilot_d657bfdf72f5/`. Trong ZIP có `summary.csv`, `cases.jsonl`, `visibility.json`, log audit số học, và SQLite. Cùng notebook session có thể chạy lại cell để tiếp tục từ SQLite sau các ca đã lưu.

- `reached_anchor / anchor_cases` đo tỷ lệ verifier đi qua bridge. `whole_anchor_accepted` đếm các ca chấp nhận cả anchor.
- `reached_all_masks_correct / reached_mask_cases` đo điền đúng tất cả mask trong các ca đã tới anchor. `correct_tokens_damaged` đếm vị trí vốn đúng bị mở rồi điền sai.
- `mean_delta_vs_hard` và `mean_delta_vs_fresh` so trên cùng các ca. `wins` và `losses` giúp xem mức tăng có tập trung vào một vài ca hay không.
- `mean_draft_ms`, `mean_guarded_verify_ms` và `mean_offline_prefix_replay_ms` tách các phần thời gian. Guard chạy thêm Qwen tuần tự. Prefix replay phục vụ benchmark từng round độc lập. Các số này mô tả chi phí pilot; tuyên bố speedup cần chạy toàn prompt với cache online và placement thực.
- `visibility.json` đo thay đổi logits khi thay một token bên phải trong cùng block hoặc ở block sau. Ca có bất đồng giữa commit Qwen thật và reference được lưu trong `numerical_failures.jsonl` và loại cả cặp khỏi summary.

Đã kiểm tra CPU hash input, hình học 48 ca, 222 ánh xạ mask và pipeline checkpoint với drafter/verifier giả lập. Đo model pretrained, acceptance và latency sẽ thực hiện trong lượt chạy Kaggle.
