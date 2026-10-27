# Draft64 → verify bình thường

Phạm vi đã chốt: **100 MATH-500 +100 GSM8K**, seed42, model/dataset revisions trong source-lock.json. Output tối đa256token/câu hoặc EOS. Không cần checkpoint thí nghiệm cũ để bắt đầu.

## Bản sửa sau mismatch FP16 ở math500:327

Chế độ hiện tại là **canonical_guarded** để thu dữ liệu correctness/headroom. Mỗi vòng vẫn chạy verifier song song, sau đó dùng chính model target và một canonical KV cursor để kiểm tra từng token trên đường sắp commit. Cursor bắt đầu từ original prompt rồi chỉ tiến bằng token argmax thật của target; nó không nhận reference tương lai. Trước commit, nếu hai đường tính khác nhau, kết quả actual canonical KV quyết định acceptance/correction. Parallel feedback gốc vẫn được giữ nguyên trong record riêng.

Guard này thêm serial target forwards, nên bản sửa **không phải benchmark SD chỉ một target forward mỗi vòng**. Các extra forwards, bất đồng token và margins được xuất ra summary.json/numerical_disagreements.csv. Vẫn giữ lượt canonical audit độc lập sau mỗi câu để phát hiện mọi lệch còn lại; mismatch không bị đổi thành pass.

Nguyên nhân trực tiếp trong ZIP lỗi: tại output53, drafter đề xuất264=` a`, nhưng parallel target false-reject và trả279=` the`; canonical target chọn264. Margin parallel top1–top2=.03125. Canonical acceptance của round đó là20, parallel chỉ16. Dữ liệu ủng hộ numerical shape/cache-path sensitivity; chưa có GPU replay để xác định kernel cụ thể hoặc margin canonical của trường hợp gốc.

## Vòng sinh

1. Từ prefix đã commit, Fast-dLLM sinh một proposal mới dài **64 token**. Mỗi unmask-forward chỉ điền một MASK, tổng64lượt; prefill/cache-update được đếm riêng.
2. Verifier AR kiểm tra proposal bằng causal parallel forward, lấy longest accepted prefix và correction khi reject; nếu nhận hết proposal thì lấy bonus token thật.
3. Commit các token đó, bỏ phần draft còn lại, rồi sinh một proposal64mới. Không cố định anchor, không kéo dài suffix cũ và không dùng retrieval để chọn output.

Active subblock vẫn **8**, physical block vẫn **32**. 64 chia hết cho8và32, nhưng prefix thực tế có thể lệch biên: một cửa sổ64token không nhất thiết bao gồm đúng8subblocks hoặc2physicalblocks đầy đủ. Không pad committedprefix để ép alignment; preflight kiểm tra phases0/1/7/8/31.

Round cuối vẫn sinh đủ64MASK dù outputbudget còn ít. Khi draft xuất hiện EOS, native scheduler vẫn hoàn tất64lượt rồi cắt logicalproposal quaEOS đầu tiên đểverify; không recycle postEOS. Correction/bonusEOS dừng output ngay. Accepted count trên proposal và emitted output được ghi riêng.

## Hai GPU và kiểm tra correctness

- Fast-dLLM-v2-1.5B FP16 trên GPU1.
- Qwen2.5-7B-Instruct FP16:18layersGPU0/10layersGPU1, không CPU/disk offload hoặc INT8. Không load AR1.5B drafter.
- Cold causal verifier prefix mỗi vòng; không dùng mutable draftKV làm targetKV.
- Trước Main: target repeatability/causal-tail audit64, ghi parallel-vs-canonical disagreements, actual guard-vs-independent-canonical test và dLLM64 cached/full equality qua5phases. Guarded mode yêu cầu guard pass; không dùng parallel-equality trên một prompt làm bảo đảm cho toàn bộ200câu.
- Sau mỗi câu: target-only canonical greedy generation độc lập từ originalprompt, tối đa384token (=256+128lookahead), dừngEOS. Kiểm toàn bộ onlineoutput và acceptedlength từng vòng. Nếu numerical mismatch thì dừng, giữ evidence; không thay output bằng reference.

Canonical reference được tạo **sau onlinegeneration**, không đi vào quyết định draft/verify. Chưa có pretrained T4x2 run của bản64 tại lúc phát hành. Tiny random CPU tests chỉ xác nhận cơ chế, không xác nhận pretrained GPU quality/speed.

## Phân tích offline

Sau reject tại a: old suffix là proposal[a+1:], prefix mới thêm correction. So old với **63token target sau correction**; loại EOS/postEOS và lưu censoring/terminal/empty-suffix riêng.

Tìm exact contiguous runs>=4token trong nonzero shifts **±8/16/32/63**. Lưu toàn bộ maximalruns, longest, chain cùng thứ tự/không overlap, bridge trước đoạn đầu và giữa các đoạn. Chain tối ưu tổnglength trong tập maximalruns đã liệt kê; không gọi là global edit optimum. Summary có sốevents>=2/3blocks, length/shift distributions, C0, bridge lengths/content và offset changes. Rounds cùng một prompt tương quan.

Các shifted matches là oracle labels cho nghiên cứu, không phải anchors trong model hoặc policy retrieval đã chạy. Chưa có pairedB24/B60 control tại cùng contexts; không quy mọi khác biệt với thí nghiệm cũ cho riêng draftlength.

## Output và resume

Mặc định `/kaggle/working/long64_guarded/long64_checkpoint.zip`, gồm SQLitebackup, promptplan/pins/runtime/preflight, rawrounds, references, diagnostics, traces/confidences, targettop1+bonus/logitmargins, parallel-vs-guard disagreements, summary/runs/bridges CSV, JSON và casesHTML.

SQLite lưu mỗi vòng; ZIP mỗi180giây tại roundboundary và sau mỗi câu/final. Dọn unusedCUDAcache cả2GPU mỗi vòng. Log trực tiếp+heartbeat15giây. Softbudget6giờ mỗi childsession; khi tới budget sẽ lưu đểresume, không đảm bảo toàn bộ200câu xong trong một session. Operational wall bao gồm peertransfers/setup/audits, không dùng để claim modelcompute speedup.

Rerun cùng cell trong cùng session để tiếp tục localcheckpoint. Phiên mới upload ZIPB64 **cùng revision và verification_mode** hoặc unzip nó và đặt LONG64_RESUME tới path đó. Giữ nguyên count/outputcap và GPU/package/tokenizer identity. ZIPB64 cũ bị lỗi ở revisionb1d0b34e, ZIPB60/Test4/pilot không phải checkpoint tương thích với bản guarded; bắt đầu mới và giữ ZIP cũ làm evidence.
