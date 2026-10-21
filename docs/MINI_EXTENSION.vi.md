# Mini-test: dLLM -> AR, anchor-conditioned redraft + extension

## Phạm vi đã chốt

- 10 rejection events từ 10 prompts trong pilot checkpoint đã unzip tại `/kaggle/input/datasets/ainzkhail/checkpt`.
- Cả draft nguồn và redrafter: `Efficient-Large-Model/Fast_dLLM_v2_1.5B`. Verifier: `Qwen/Qwen2.5-7B-Instruct`. Không tải hoặc chạy AR drafter.
- 5 GSM8K +5 MATH-500; 6 enriched +4 representative; first rejection1-based8–24; E=8. Seed42, chọn từ metadata/cohort, không chọn theo kết quả repair cũ/mới. Nếu không đủ quota thì dừng, không tự nới điều kiện.
- Tham chiếu model/source revisions giống pilot; FP16, SDPA, target chia18/10layers trên2GPU, dLLM trênGPU1. Không quantization/offload/fallback.
- Physical32, earliest ABSOLUTE active subblock8, mỗi forward unmask CHÍNH XÁC1MASK. Đây không phải global-H confidence scheduling.
- Mini-test chỉ đo chất lượng proposal; không báo latency, speedup hay throughput. Forward counts tách primary/composed, newly executed và cached/full audit.

## Ma trận 9 arms/event

| Strategy | L | H Joint | H Sequential |
|---|---|---|---|
| Fresh | fresh_L | fresh_H_joint | fresh_H_sequential |
| Oracle | oracle_L | oracle_H_joint | oracle_H_sequential |
| Predicted | predicted_L | predicted_H_joint | predicted_H_sequential |

Tổng90 logical arms trước aliases. Joint khởi tạo mọi vị trí không KEEP trong H bằng MASK và chạy lịch native strict1token. Sequential dùng lại kết quả short arm cùng strategy, khóa TOÀN BỘLtokens làm prefix, rồi sinh thêm8tokens; không sửa lại phầnL.

Mỗi unique generated stage chạy cached và full-prefix independently; bắt buộc tokens bằng nhau. Actual AR verifier kiểm mỗi unique final proposal. Prefix/reference disagreement dùng actual-model canonical replay từ original prompt như recovery pilot; không thay logits bằng gold. Đây là offline diagnostic, không phải online SD.

## Anchors đóng băng

Budget B=floor(L/4), tối đa2 contiguous spans, mỗi span>=2tokens; không dùng singleton fallback. Không đủ span thì mask rỗng; không ép fill budget.

- Oracle: old tokens khớp corrected reference; ưu tiên span dài nhất, sau đó vị trí sớm nhất; cắt bằng cách xét mọi subspan dài2..B. Không phải optimal acceptance oracle; không inject gold tokens mới.
- Predicted: old token bằng old verifier top1, old-token probability>=0.8, drafter confidence>=0.5. Score span bằng trung bình `0.5*(log p_verifier + log p_drafter)`, tie theo dài hơn rồi sớm hơn. Greedy chọn tối đa2spans không chồng/không kề nhau và không vượtB. Selector không nhận reference.
- Giữ cùng anchors trong L/Joint/Sequential. Mask trùng positions+values trong cùng event/schedule/horizon là aliases, không đếm như independent trials.

## Dùng lại checkpoint và những kiểm tra còn cần

Nhập read-only `checkpoint.sqlite` cùng `run-manifest.json` từ completed recovery pilot `fa330e5836db310172ced163162d511f6b2036b5`; từ chối identity/source/config không tương thích, unresolved failures hoặc manifest/DB sai nhau. Giữ SHA256 input, exact frozen event IDs/prefixes/anchors/gold và model/environment identity.

Bỏ qua toàn bộ target reference generation, Test1 và Test2 pilot. Chọn events có reference đủ tới `offset+40`, không cần sinh kéo dài mới; không vượt EOS. r=k+1: corrected prefix chốt k accepted tokens + correction; old suffix là draft[r:], L=32-r.

Fresh-L chạy lại đúng10events để có unmask order/trace, vì Test2 pilot không lưu hai trường này. Bắt buộc output token IDs bằng Fresh-L pilot tương ứng; disagreement dừng với evidence. Không giả mạo trace bằng trace Test1.

Session mới vẫn phải tải/nạp hai model, kiểm tokenizer và numerical environment so pilot, target repeatability/parallel/causal preflight. Không dùng suite preflight pilot có AR drafter. Trong quá trình chạy, mỗi unique dLLM stage được audit cached/full trên chính geometry thực.

H<=32 không bảo đảm single physical block. Ghi phase, absolute block/subblock, prediction-row/carry, extension-before-old-complete và actual future visibility. MASK padding vốn tồn tại trong short arm; không diễn giải extension như tự động thêm bidirectional context.

Kết quả save mỗi arm; chạy lại cùng cell/output bỏ qua arms đã lưu, dùng đúng frozen identity. Clear unused CUDA cache cả2GPU sau mỗi event; các KV của mutable branch không tồn tại qua arm. Đóng gói partial ZIP cả khi input/setup runtime/preflight/inference thất bại nếu output còn ghi được. Setup pip log nằm riêng ở `/kaggle/working/extension_setup.log`.

## Output

Mặc định `/kaggle/working/extension_mini/extension_checkpoint.zip` (nếu revision khác đã có output, dùng thư mục có suffixrevision để bảo toàn dữ liệu).

- `inputs.json`, `extension-manifest.json`, `runtime.json`, `preflight.json`: frozen design/input/model/checks.
- `checkpoint.sqlite`, `arms.jsonl`, `arms.csv`: toàn bộ90 logical arms/aliases, tokens/reference, hardKEEP, actual trace, first mismatch, acceptance và forward counts.
- `paired.csv`: ΔA so Fresh cùng horizon VÀ schedule, ΔA so short arm, win/tie/loss và ceiling flags.
- `events.csv`: tất cả10events, cácarm và schedule interaction.
- `REPORT.md`, `summary.json`, `cases.html`: tổng hợp và token-level visualization3best+3worst theo max Oracle gain so schedule-matched Fresh-H. Nếu xem lại completed output không nạp tokenizer, HTML dùng tokenIDs.
- `execution_status.json` và evidence khi lỗi. Chỉ `status=complete` với90/90arms mới hoàn tất.

Không lấy 10purposive events để ước lượng hiệu quả population; first mismatch ở REDRAFT/EXTEND là vị trí thất bại, chưa phải bằng chứng nguyên nhân. Chưa có pretrained GPU mini-test pass trước khi người dùng chạy cell và trả ZIP.
