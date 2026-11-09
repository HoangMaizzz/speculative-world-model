# SAVH · Pilot source-span detection

Pilot kiểm tra giá trị bổ sung của hidden layer 27 và correction token trên cùng mô hình. Dữ liệu đầu vào là verifier features B64 đã hoàn tất. Runner dùng ZIP gốc, thư mục giải nén có events.jsonl và shards/, hoặc thư mục chứa 58 ZIP đã chia. Metadata, shard được dùng và tensor được dùng đều phải khớp SHA-256. Không cần Long64 ZIP hoặc tải Qwen/Fast-dLLM. 

## Giao thức

- Giữ prompt split 164/19/17. Full đọc mọi event có correction đã emit (3.755 trước khi lọc cửa sổ đủ dài). Nhóm corrected cold mismatch không bị loại chỉ vì corrected replay, vì pilot chỉ dùng old-path features. Test cũ là exploratory.
- Nhãn exact 4/8 từ token IDs với i khác j. Any-shift trong cửa sổ quan sát, near-shift tại j=0..7. Hai mục tiêu fit và báo cáo riêng. Cửa sổ ngắn hơn L được loại. Mỗi window luôn nằm trong một span liên tục khớp reference.
- HGB scalar và MLP factorial scalar, scalar+correction, scalar+hidden, scalar+hidden+correction. Correction là cặp rejected/corrected token ID, embedding train-only với ID chưa gặp ánh xạ về 0. Hidden là post-read layer 27, PCA32 train-only sau L2 normalization. MLP đọc từng vector theo thứ tự trong cửa sổ. Đây là frozen PCA pilot, không phải Span Transformer nhiều layer.
- Scalar gồm top1, logp, rank, gap, entropy, margin, drafter confidence, vị trí, tần suất token và norm hidden. Kiểm order[position]=step từ trace, không đảo mảng order. Cùng scalar cho mọi nhánh. Không đưa q0, corrected hidden, M_off, destination labels hoặc future vào input. 
- HGB seed 42, MLP seeds 42/123/2026, AdamW, BCE, tối đa 12 epoch. Chọn epoch theo validation AP. Một cửa sổ điểm cao nhất mỗi event. Ngưỡng validation theo coverage 2/5/10/20/50%, thêm precision 90/95% nếu có ít nhất 12 lượt. Báo coverage thực, exact precision, recall, shortlist recall@8, token capture và bootstrap theo prompt.
- Không dùng các kết quả này để khẳng định speedup hoặc detector đặt đúng destination. Chỉ sau khi hidden có tín hiệu trên near-shift mới đáng mở rộng mô hình hoặc thử actual redraft.

## Chạy và lấy kết quả

`python -u scripts/train_savh_pilot.py --input /path/to/checkpoint --output /path/to/output`

Thêm `--smoke` để kiểm 72 events trải đều qua ba split, fit đủ 5 nhánh với L4 near, một seed. Full dự kiến 52 fits. Có thể tiếp tục cùng output sau khi gián đoạn. Runner dùng cache feature và bỏ qua fit đã có báo cáo. Dùng thư mục riêng cho smoke và full.

Kaggle GPU T4 x1 đủ cho head. Bật Internet để lấy runner. Gắn dataset có ZIP gốc hoặc đủ 58 ZIP, sau đó trỏ INPUT chính xác. Trình huấn luyện dựa trên NumPy, scikit-learn, PyTorch, PyArrow và safetensors. Chỉ head dùng GPU. Chuẩn bị khoảng vài GB trống trong working để chiếu layer 27. Dữ liệu gốc đọc theo shard và memmap FP16 được xóa sau khi hoàn tất projection.

`status.json` xác nhận hoàn tất, `summary.json` chứa metric mỗi fit. Các file `*_scores.npz` ghép với `input_*.npz` và `events.json` để tái lập dự đoán từng cửa sổ. Lưu PCA, scaler, vocabulary và weights của head. Thời gian ghi trong báo cáo là thời gian fit và scoring, không phải latency online.
