# Thu verifier features cho shifted suffix — full B64

## Input và phạm vi

Tái dùng checkpoint guarded B64 hoàn tất: 100 GSM8K +100 MATH, 4050 rounds, 3847 eligible rejection events (1841 GSM/2006 MATH). Source revision `39890af682c068a51f5788594195f86e79a28575`. ZIP đã audit SHA256 `fb7e1ea13a0b6f6fce9f20d22c5270ab2c54e70cf83c59f7b95b31b44cacafca`. Config khóa SHA256 của sáu exported input files để dùng được cả ZIP và folder unzip.

Qwen/Qwen2.5-7B-Instruct revision a09a35458c702b33eeacc393d103063234e8bc28, FP16, SDPA, split18/10 layers trên 2 Tesla T4, batch1; torch2.9.0+cu126, transformers4.53.3, accelerate1.10.1 giống input. Không load Fast-dLLM, không regenerate drafts/references, không repair/anchor, cây hay detector training.

Một cell độc lập tự setup, CPU tests, input audit và labels, load Qwen, smoke20 -> pilot500 milestone -> full3847. Smoke và pilot thuộc full, không thu lại. CLI `--limit 20` hoặc `--limit 500` chạy riêng và cùng config có thể resume full. Soft budget mặc định6h sau load model; nếu hết budget thì lưu partial ZIP và tiếp tục phiên sau. Thời gian cài đặt/download, tổng cell wall và inference budget khác nhau.

## Indexing

Reconstruct X từ prompt_ids trong prompts.jsonl và committed_tokens của các rounds trước; kiểm index/offset/prefix hash/commit/EOS/native64 trên toàn4050 rounds. Reference chỉ dùng offline audit/labels.

Native draft D64 khác logical proposal V đã cắt sau EOS. 250 rounds có len(V)<64; 175 eligible. a/c từ decision authoritative sau guard. S=V[a+1:] cắt trước EOS. X_corr=X+V[:a]+[c]. Source index i tương ứng proposal a+1+i và absolute P+a+1+i. Prediction V[t] tại P+t-1; post-read hidden tại P+t.

92 eligible events có correction được guard xác định nhưng output cap cắt trước correction: trường `correction_emitted_within_output_budget=false`. Giữ nhóm này để chẩn đoán, không gọi là correction đã emit trong một deployment.

Old replay dùng đúng X+V, use_cache=False, logits_to_keep=len(V)+1. Đối chiếu `verification.parallel_feedback`, không dùng feedback top-level đã trộn canonical guard. Giữ riêng14 stored guard disagreement rounds và mọi replay mismatch mới.

## Features mặc định

- Tất cả proposal positions: draft logp, rank =1+count(logit>draft logit), top1-minus-draft logit gap, greedy match/tie flags, entropy toàn vocabulary, top1 probability, top16 IDs và full-vocabulary-normalized logp. Bonus có distribution riêng, không có draft-token features.
- Sparse hidden outputs decoder block IDs zero-based7/14/21/27 trước final RMSNorm. Old hidden lưu union positions P+a..P+a+L; slot i trước dự đoán S[i], slot i+1 sau đọc S[i]. Không lưu cả prompt hidden hoặc output_hidden_states toàn bộ layers. FP16 CPU storage.
- M_off[i,j] với j=0..L: logits local a+1+j trên đường proposal cũ; j=L là prediction ở biên sau source cuối (có thể là EOS/bonus). Source i và column j được lưu riêng validity mask. Ma trận không đại diện corrected context.
- Corrected cold full-prefix forward: hidden selected layers sau correction, final normalized h_c, next top16/entropy/margin, FP32 q0_logp_source. q0 chỉ chấm immediate next position; token ID lặp có q0 giống nhau.
- Cosine h_c với old pre-predict và post-read states riêng. Đây là feature thăm dò, không là calibrated alignment probability.
- Native drafter tokens/confidence/order/trace đọc lại từ ZIP; confidence ở lúc token được unmask, không phải final-draft probability.

Top32 và layer subset qua CLI. Logits/rank/entropy reduction bằng FP32; không lưu full-vocab logits. Giữ hidden FP16, score/matrix FP32. Bốn lớp old suffix post-read đã5.36GiB raw; union slots và corrected states cộng thêm. Cần khoảng14GiB working free ở đầu để chứa shards và ZIP tổng hợp; output thường vài GiB, phải đo smoke.

## Optional micro-probe

Default `--micro-probe-steps 0`. Với1..3, lấy greedy token thực từ corrected context và tìm mọi occurrence từng probe prefix trong S. Token đầu đã có từ corrected logits; token2/3 thường cần thêm1/2 target KV forwards. Dừng probe khi EOS. Lưu positions/competing matches/tail candidates, calls và timing. Không đọc reference để mô phỏng probe; probe tokens không được tính là token cũ reuse.

## Offline labels và kiểm tra

offline_labels.parquet giữ mọi maximal exact run>=4, zero và nonzero delta=j-i, mọi start-pair match4/match8, ambiguity, initial bridge tới từng run và edges giữa các maximal runs không chồng lấn theo cả hai trục. Đây là cạnh của graph maximal runs, không liệt kê mọi chain/partial-run optimal alignment. Giữ source/reference endpoints, EOS và thiếu reference riêng. 14 rejection diagnostics không đủ điều kiện bị loại có lý do, không gán negative.

Smoke20 chọn cố định, enrichment theo dataset/a/short-proposal/C0/multi-run-compatible/negative/repetition/stored guard disagreement. Pilot round-robin dataset và acceptance bins; toàn3847 không enrichment. Split train/validation/test dùng hash prompt, không round. Chưa train và chưa đo detector accuracy.

Smoke kiểm capture vs plain output, thay token cuối proposal để kiểm causal tail, indexing/tensor shape/finite values/rank/logp/topK. Old Top1 mismatch trên smoke -> giữ evidence và dừng full. Trong full, replay mismatch được giữ raw + quarantine flag; hoàn tất sẽ có status complete_with_quarantine. Corrected cold-vs-cached-canonical next mismatch chỉ là audit riêng; không thay logits bằng reference. GPU/backend differences có thể có dù cùng precision; không hứa bitwise parity cho mọi path.

## Output, độ bền và resume

- feature-manifest.json: config/input hashes/code/source digest/model/environment/schema.
- event-plan.json: frozen smoke/pilot/full IDs, attrition.
- shards/part-xxxxx.zip: immutable batches32 events, records.jsonl và tensors/*.safetensors. Smoke được flush ở20. Hash từng tensor payload và từng shard. Không pickle.
- events.jsonl: online scalar/topK/native trace features và IDs, không oracle labels.
- replay_checks.jsonl: old replay checks và evaluation-only reference audits.
- timing.csv: phase/model stage compute ms không peer dispatch; wall forward/capture và reduction wall riêng. GPU reduction/capture/CPU I/O không nằm trong model-stage compute sum. Không claim speedup từ collector timing.
- offline_labels.parquet, shards-index.json, smoke-validation.json, summary.json, report.md, execution_status.json.
- verifier_features_checkpoint.zip: chứa metadata, labels và mọi shard ZIP, đóng lúc kết thúc/budget/failure. Shards đã nén nên ZIP tổng dùng STORE. Thiếu disk đóng ZIP thì giữ các shard riêng và warning.

Atomic flush mỗi32events và tại smoke/pilot/final/error. Khi kill cứng có thể mất batch đang nằm trong RAM tối đa31events; shard đã rename thành công được verify rồi skip. Chạy lại cùng output tự resume; phiên khác đặt `--resume-root` vào ZIP output hoặc folder đã unzip. Cần source B64 input gốc cùng resume features. Resume từ chối khác code/config/input/environment/corrupted tensor; không merge các package probes/layers khác nhau. Model buffers được bỏ tham chiếu sau event, CPU batch clear sau flush, CUDA unused cache clear tại shard boundaries.

## Commands

Sau cài requirements và `pip install --no-deps -e .`:

```bash
python scripts/audit_verifier_feature_input.py --input-root /kaggle/input --output /kaggle/working/feature_input_audit
python -m unittest discover -s tests -p 'test_verifier_features*.py' -v
python -u -m suffix_lab.verifier_features.run_collection --input-root /kaggle/input --output /kaggle/working/feature_smoke --revision YOUR_FULL_COMMIT_SHA --limit 20
python -u -m suffix_lab.verifier_features.run_collection --input-root /kaggle/input --output /kaggle/working/feature_pilot --revision YOUR_FULL_COMMIT_SHA --limit 500
python -u -m suffix_lab.verifier_features.run_collection --input-root /kaggle/input --output /kaggle/working/feature_full --revision YOUR_FULL_COMMIT_SHA
```

Smoke20/pilot500/full không cần output folders riêng; dùng cùng folder sẽ tái dùng completed shards. `--resume-root /kaggle/input/FEATURE_DATASET` dùng sau khi đổi session. `--layers 14 27 --top-k 32 --micro-probe-steps 2` tạo package khác, cần output mới.

## Validation đã thực hiện

CPU audit input thật4050rounds/200prompts/3847eligible,200581source tokens,18116maximal runs gồmzero,17182nonzero,23709compatible bridge edges. Có bộ tiny random CPU Qwen tests cho actual logits/hidden/probe, no-future-label input, sparse EOS, archive restore và skip completed. Local tests không chứng minh pretrained FP16 T4 parity, latency hay detection performance; cell thực hiện smoke trên GPU trước full.
