# Test 4 — Phase 1: Main 200, EOS-aware suffix recycling

Triển khai bản đặc tả người dùng chốt ngày 21/10/2026. Chỉ Phase 0 correctness và Phase 1 Main; chưa chạy stress coverage/position/precision, Sequential hay huấn luyện selector.

## Phạm vi

- 100 GSM8K + 100 MATH-500, đúng một eligible event/prompt.
- Ba arms/event: `F-23`, `O-All-Joint-23`, `KEEP-All-23`; tổng 600 logical arms.
- Fast-dLLM v2 1.5B FP16 draft/repair; Qwen2.5-7B-Instruct FP16 verify, GPU0 18 layers/GPU1 10 layers. Không nạp AR drafter.
- Native physical32, earliest absolute subblock8, strict highest-confidence Top1: mỗi unmask forward điền đúng một MASK. Giữ cơ chế right-shift/carry và cache chỉ completed physical blocks của Test3.
- Không nghiên cứu latency trong Phase1: chỉ lưu forward/input-token counts, không xuất thời gian model có peer-transfer như số đo khoa học. Session wall time phục vụ checkpoint/budget, không là speedup.

## Input và chọn mẫu

Input là **pilot FP16 đã hoàn tất**, đã dùng cho Test3, tại `/kaggle/input/datasets/ainzkhail/checkpt`; không phải ZIP Test3. Kiểm manifest/runtime/source/numerical identity và SQLite integrity; copy/hash tuần tự sang ổ tạm trước khi đọc. Dữ liệu input giữ read-only. Không tái sử dụng proposal B32 hay 10 events đã chọn có chủ đích của Test3.

100 primary prompts/benchmark là đúng tập prompt trong pilot, thứ tự index tăng dần được freeze. Tải toàn bộ test split ở dataset revisions trong `source-lock.json`, kiểm primary question khớp, tạo reserve list gồm mọi prompt còn lại theo deterministic seeded shuffle. Lưu toàn bộ `prompt_plan.json` trước screening.

Mỗi prompt có đúng 5 offsets cố định `[0,56,111,167,223]`. Không thay offset theo Oracle/Fresh hoặc lùi offset khi gần EOS. Các offset sau EOS/context limit được ghi attrition. Các offset này là reference-prefix sampling, không phải phân phối visitation của online SD.

Mỗi offset còn hợp lệ được chạy native **B24 thực sự**, không cắt B32. Kiểm tất cả 5 offsets rồi chọn đều ngẫu nhiên trong tập eligible của prompt bằng RNG từ `(seed42,prompt_id)`. RNG không phụ thuộc thứ tự resume hoặc kết quả repair. Không chạy Fresh/Oracle để chọn event. Không đủ eligible thì chuyển sang reserve tiếp theo; cạn reserves báo `selection_shortfall`, không giảm quota. Freeze cả 200 events trước khi chạy bất kỳ Main arm nào.

## Eligibility và EOS

Tập stop IDs được đọc từ tokenizer EOS và generation config của hai checkpoint, lấy union cố định, lưu từng nguồn và effective IDs trong runtime. Đây là quy tắc termination áp dụng đồng nhất, không dò decoded text.

- `a=0..23` đều được lấy. `L=23-a`, `E=a`, capacity H=23 sau correction. Correction đã thuộc committed prefix, không tính vào acceptance mới.
- Accept đủ B24: loại `all_24_accepted`.
- EOS trong old draft trước vị trí cuối: loại `early_draft_eos`. Ưu tiên ghi nhóm này trước verification; terminal-correction count là trong các states đã qua bước loại early draft EOS.
- EOS chỉ ở vị trí cuối B24 vẫn giữ fixed-window geometry, nhưng token EOS đó trở lại MASK, không làm lexical KEEP trong bất kỳ arm nào.
- Correction là EOS: loại `terminal_correction`; không tạo lượt tiếp theo.
- Target EOS trong 23 continuation positions: **giữ**, báo `target_terminal_within_horizon`. Không dùng endpoint tương lai này để rút ngắn drafter capacity.
- Reference thiếu vì token budget: nối từ original prompt, replay từng committed token bằng actual argmax rồi tiếp tục greedy. Không re-prefill toàn bộ long prefix để tạo trajectory khác; không thay logits bằng saved gold. Numerical replay mismatch là lỗi phải dừng.
- Context limit làm thiếu gold: ghi `reference_censored`; không dùng làm eligible full-comparison event. Runtime/model error dừng và lưu failure, không biến thành accepted length0.

Native dLLM **vẫn hoàn thành tensor 24/23 positions**, kể cả khi EOS được unmask trước một số MASK phía trước. Sau đó cắt logical proposal tại EOS đầu tiên; verifier chỉ nhận logical tokens và không vượt target EOS. Các token sau EOS trong tensor không được verify/commit/recycle. Ghi số native unmask forwards đã thực hiện ở các vị trí bị cắt; không claim early-stop compute savings. Không sửa attention hay scheduler theo future target EOS.

Oracle chỉ KEEP token có thật trong old suffix, khớp reference ở vị trí trước target EOS; không KEEP EOS hoặc token sau endpoint. KEEP-All dùng mọi token thật trong old suffix trừ configured EOS, **không xem future target EOS để chọn mask**. Các old tokens vượt target endpoint ở KEEP-All có thể nằm trong native canvas nhưng không được chấm sau EOS; đây là khác biệt có chủ đích giữa non-oracle action và oracle labels.

Mỗi proposal được actual target reverify độc lập. `checked_score` giữ parallel evidence, dùng canonical one-token replay khi parallel/reference bất đồng, và hard-fail nếu replay vẫn bất đồng. Gold-triggered fallback là **offline diagnostic**, không là online SD/speedup protocol.

## Correctness / fingerprints / resume

Phase0 trong cell chạy EOS/selection/checkpoint unit tests trên CPU trước khi nạp pretrained models. Sau đó real-GPU preflight kiểm target repeatability/parallel/right-shift/causal-tail và dLLM cached/full + hard KEEP ở physical phases0,1,7,8,31.

Mọi native arm kiểm KEEP immutability, length/order, mask delta1, `unmask_forwards=unmask_steps=H-KEEP`. Mỗi event có full gold continuation audit bằng actual target. Selected B24 và cả ba Main arms được cached/full audit ở mỗi event thứ10 (gồm đầu mỗi benchmark); strict invariants và actual verification vẫn chạy ở mọi unique arm.

Event fingerprint gồm prompt/prefix/original draft/corrected prefix và model revisions. Arm fingerprint gồm event fingerprint, KEEP positions/values, horizon, schedule, L, toàn bộ decoding config, model và code revision. Tên human-readable không tạo action khác: Oracle không có anchors có thể alias Fresh; equal acceptance đơn thuần không được alias.

SQLite synchronous commits sau mỗi reference/state/arm; event và prompt decision chung một transaction. Portable ZIP được refresh khoảng mỗi120 giây tại ranh giới công việc, lúc tạo reference, khi freeze cohort và trong finalization. SIGKILL có thể mất phần sau lần ZIP gần nhất nếu chỉ mang ZIP sang phiên khác; SQLite tại working directory vẫn có các unit đã commit. Kaggle soft wall budget6h dừng ở unit boundary với `budget_checkpoint`, không gắn nhãn complete. Setup/pip trước child process không nằm trong6h; không bảo đảm hoàn tất một session.

Cell có `resume = None`. Muốn sang phiên mới, attach ZIP Phase1 (hoặc thư mục đã unzip) và đổi thành đường dẫn input đó. Vẫn giữ pilot input. Resume chỉ nhận đúng identity/runtime/config; restore chỉ checkpoint SQLite đã integrity-check, không extract tùy ý ZIP. Không chạy lại sampling plan, selected events hay arms đã lưu; runtime preflight vẫn chạy lại mỗi phiên chưa complete.

## Báo cáo

- `A_inclusive`: số draft tokens accept, bao gồm EOS đúng khi đã khớp mọi token trước đó.
- `A_content`: số accepted tokens trước EOS; `I_EOS`, `terminated_correctly` riêng.
- W/T/L, mean/median paired ΔA và95% prompt bootstrap CIs (2000 resamples), mỗi benchmark riêng.
- Bốn strata: all eligible; reuse-applicable (`L>0,C>0`); nonterminal full-horizon; target-terminal. Không gộp terminal ceiling với full23 rate.
- Attrition: prompts/states/native B24 examined, reserves, exclusions, censoring, terminal correction, early draft EOS; histograms a/L/E/C và C0.
- Logical token IDs, accepted extension, mutable wrong→right/right→wrong theo REDRAFT/EXTEND; không so token ở vị trí sau logical EOS. Gain attribution tách direct KEEP, bridge đã đúng và mutable corrected.
- Anchor raw pieces có thứ tự ưu tiên: special/control; newline/tab/indentation; whitespace-only; punctuation; numeric; lexical/mixed. Không `strip()` để phân loại token dấu cách; decode riêng ID220 bằng đúng checkpoint tokenizer.
- `summary.json/csv`, `events.csv`, `arms.csv`, `paired.csv`, `report.md`, `cases.html`, `main_distributions.png`, `anchor_tokens.json`, `attrition.json`, JSONL, SQLite và provenance trong ZIP.
- Report partial chỉ tổng hợp các events đã đủ cả3 arms, có N/complete flag rõ; không gọi partial là Main200 hoàn tất.

Local unit/mocked/tiny-random checks chỉ kiểm mechanics. Chưa có pretrained GPU evidence của Test4 khi công bố cell này.
