# Suffix Reusability & Selective Redrafting

**Protocol v4 (20/10/2026):** Test2 tách **representative** và **repairability-enriched**, disjoint và freeze trước inference. Mẫu representative cân bằng source/benchmark/progress strata, uniform trong mỗi stratum, lưu inclusion probability và inverse weight; chỉ nhóm này dùng ước lượng population hữu hạn. Enriched chỉ chọn suffix còn token cũ đúng, ưu tiên matched islands dài/count cao trong strata; báo riêng, không gộp làm hiệu quả thực tế. Không đủ eligible events thì báo shortfall, không bù bằng suffix rỗng.

Smoke mới: **5 MATH +5 GSM8K**, reference256, tối đa3prefix/prompt và60 Test1 proposals; Test2 nhắm **24 events =12 representative +12 enriched**, cả hai redrafters. Pilot giữ100+100, tối đa5prefix/2000 proposals, Test2 nhắm100+100 events. Kết quả cũ v3 pass GPU nhưng không thay thế smoke v4.

Oracle KEEP old-correct, bounded old-mask search và Future Reference Injection được đặt cạnh nhau và tách nhãn. Báo K_correct0/1/2–3/4+, vị trí/island, ΔA/ΔT, false-KEEP caps, equal-quality drafting savings và accepted tokens/ms cùng verification cost. Future-anchor ablation có3 seeds/layouts trên context lengths/progress thực tế; mỗi seed giữ cùng positions/count cho old/correct/wrong/shuffled/random và cả hai redrafters. Không coi repeat/alias là event độc lập.

Output bổ sung: `sampling_coverage.json`, `analysis/sampling_membership.csv`, `representative_population_estimates.json`, `oracle_comparison.json`, `benefit_outcomes.csv`, `efficiency.csv`, `summary.csv` và `conditioning.csv`. Phần ước lượng population chỉ xuất point estimates design-weighted khi có đủ stratum/outcomes; bootstrap trong summary mô tả từng cohort và không phải population CI.

Vòng đầu: **100 MATH-500 + 100 GSM8K**, seed 42. Tối đa 5 shared prefixes/prompt, 32 candidate tokens, tối đa **2.000 Test 1 proposals**. Test2 nhắm200 rejection events có suffix không rỗng:100 representative +100 enriched; cân bằng source/benchmark/progress khi đủ dữ liệu và report rejection bins. HumanEval và general instruction chưa chạy trong vòng này.

Đây là hai thí nghiệm **offline mechanism/headroom**, chưa phải online speculative decoding, chưa chứng minh speedup. Không fine-tune weights. Xem protocol và giới hạn ở [docs/PROTOCOL.md](docs/PROTOCOL.md).

| Vai trò | Checkpoint | Thiết bị / precision |
|---|---|---|
| Verifier/reference | Qwen/Qwen2.5-7B-Instruct | FP16 cố định, chia lớp trên GPU0 + GPU1 |
| AR drafter | Qwen/Qwen2.5-1.5B-Instruct | GPU1 FP16 |
| Diffusion drafter/infiller | Efficient-Large-Model/Fast_dLLM_v2_1.5B | GPU1 FP16, load lần lượt với AR |

`source-lock.json` khóa model/dataset revisions, official Git revision và hash code remote. `prepare` xác minh hash trước inference. Không dùng CPU/disk offload.

Theo yêu cầu mới20/10: không dùng INT8/NF4 và không fallback lượng tử. GPU0 giữ embedding, rotary và18decoder layers đầu; GPU1 giữ10layers cuối, norm, lm_head và một drafter1.5B FP16. Test1 chạy dLLM trước, unload/GC/empty allocator caches trên cả hai GPU, rồi mới load AR. Test2 freeze cùng sample và anchor plans cho cả hai redrafters, chạy dLLM rồi unload và chạy AR constrained. Preflight kiểm hai drafter lần lượt khi verifier vẫn resident. Precision/device map/timing policy được lưu trong identity; không resume kết quả INT8 thành FP16.

Thời gian verifier chính là tổng CUDA-event durations trong các compute stages (embedding/rotary/decoder/norm/head), đặt bên trong Accelerate dispatch hooks để loại copy giữaGPU. Đây là stage compute time, không phải latency đầu cuối; không gồm Python/IO giữa các stages. `wall_latency_ms` được lưu riêng, không lấy chênh lệch wall-minus-compute làm thời gian truyền dữ liệu. Drafter chỉ ởGPU1, không có inter-GPU copy trong forward; tiếp tục báo synchronized wall/cold replay. Session budget6h luôn là wall time thực tế.

## Chạy trên Kaggle

1. Notebook Settings: **Internet ON, GPU T4 x2**.
2. Repo public: không cần GitHub token. Nếu repo vẫn private: add/enable Kaggle Secret **GITHUB_TOKEN** có quyền đọc repo. Không dán token vào cell. HF_TOKEN tùy chọn.
3. Import bản **một code cell** phù hợp: [smoke5+5](notebooks/math_gsm8k_smoke.ipynb) hoặc [pilot100+100](notebooks/math_gsm8k_pilot.ipynb). Có thể copy từng cell từ [KAGGLE_TWO_CELLS.md](KAGGLE_TWO_CELLS.md). Mỗi cell tự setup và chạy correctness checks; không cần biến `lab`, không cần smoke status trong phiên test thật.
4. Để Save Version chạy thật, dùng notebook pilot riêng chỉ chứa một code cell, chọn **Save Version → Save & Run All**. Không để cell smoke trong notebook này, vì Save & Run All sẽ chạy lại mọi cell từ đầu. Internet vàT4x2 cần bật trong notebook được lưu.
5. Cả hai bản tự xuất ZIP/report/10plots, phân biệt complete/partial/failed và khôi phục ZIP đúng mode/config/commit được attach vào `/kaggle/input`. Kết quả cũ khác code revision được giữ nguyên; launcher chọn output riêng cho revision mới. Phiên Save Version không dùng biến/cache RAM của phiên interactive.
Notebook7cell cũ vẫn được giữ để xem chi tiết từng giai đoạn. Bản một cell dùng runtime/source commit v3 FP16 cố định `c78cd1ba7da5f52b078a7537460e08f949746c39`, kiểm SHA256 launcher trước khi thực thi; không lấy code mới tùy theo branch head khi Save Version chạy sau này.

Launcher setup đã kiểm tra, giữ pin riêng `274762b1cb9e5d8058bced1d7672152748c0a9df`: hỗ trợ Kaggle Python3.13 thiếu `ensurepip`, tạo venv bằng `--without-pip`, seed pip bằng `pip --python`, và sửa venv tạo dở thiếu pip. Không cài vào Python hệ thống. Log setup được in trực tiếp và lưu tại `/kaggle/working/suffix_setup_smoke.log` hoặc `suffix_setup_pilot.log`. Kết quả preflight cũ cũng được giữ riêng khi chưa có run-manifest, dựa vào execution_status/code_revision.

Runtime723200ac sửa dependency thành `pyarrow==21.0.0`, đáp ứng yêu cầu `pyarrow>=21.0.0` của `datasets==4.2.0`. Dùng cell mới: cell pin runtimec4a25faf vẫn clone requirements cũ dù branch đã sửa. Launcher tự chọn source directory mới và cài lại requirements trong venv hiện có, không cần xóa checkout cũ. Không đổi protocol/model/dataset, nhưng run identity đổi theo code/packages; không trộn checkpoint từ pin cũ.

Runtime18d665bd giữ sửa dependency trên và sửa môi trường biểu đồ: dùng `MPLBACKEND=Agg` trước khi import Matplotlib, loại `PYTHONPATH`/`PYTHONHOME` kế thừa khỏi các tiến trình Python con. Điều này tránh backend inline/startup hooks chỉ có trong notebook bị kéo vào venv. Các test giả lập không còn in COMPLETE gây nhầm với smoke thật; chỉ tin trạng thái cuối của lần chạy. Không đổi protocol, và vẫn bắt buộc pretrained GPU preflight trên Kaggle.

Runtime90d6442a thống nhất greedy argmax trên logits, tie chọn token ID nhỏ nhất, cho drafting/verifier/dLLM. Không dùng phần tử đầu của topk làm greedy winner vì topk không bảo đảm tie order. Preflight ghi tách repeatability và parallel shifted equality; nếu fail, `preflight_failure.json` và ZIP chứa token IDs/vị trí/margins cùng full-prefix replay. Log Kaggle cũ chỉ có assertion gộp nên chưa xác định nguyên nhân chính xác; bản sửa tie đã được test local nhưng chưa có pretrained GPU pass. Cảnh báo sitecustomize/wrapt còn xuất hiện, chưa chứng minh có liên quan mismatch. Không bỏ correctness checks hay tự đổi INT8 để che mismatch.

Log sau đó xác định lỗi INT8 vẫn còn: cached token17767 khác full/parallel220 tại vị trí2, repeatable=true và margin0.625/0.0625, nên trường hợp này không phải tie. Bản hiện hành chuyển FP16 theo yêu cầu người dùng; preflight còn kiểm thay token cuối không được đổi dự đoán causal phía trước. Requirements bỏ bitsandbytes, thêm wrapt1.17.3 cho module thiếu được sitecustomize báo. Đây không phải tuyên bố wrapt là nguyên nhân lệch token hay đã chạy pretrained FP16 thành công.

Notebook dùng môi trường riêng trong `/tmp`, Torch 2.9.0 CUDA12.6 và dependency pins để tránh package đã import trong Kaggle kernel. Model cache trong `/tmp`, không nằm trong artifact. Commit thực tế, package versions, precision và GPU được ghi vào identity.

## Lệnh tương đương

Sau khi cài Torch phù hợp, `pip install -r requirements-kaggle.txt` và `pip install -e .`:

```bash
python -m unittest discover -s tests -v
suffix-lab prepare --smoke --assets assets/smoke --output outputs/smoke
suffix-lab preflight --smoke --assets assets/smoke --output outputs/smoke
suffix-lab run --smoke --assets assets/smoke --output outputs/smoke
suffix-lab analyze --smoke --output outputs/smoke

suffix-lab prepare --assets assets/pilot --output outputs/pilot
suffix-lab preflight --assets assets/pilot --output outputs/pilot
suffix-lab run --assets assets/pilot --output outputs/pilot
suffix-lab analyze --output outputs/pilot
suffix-lab pack --output outputs/pilot
```

`run --stage references|test1|test2` cho phép chia giai đoạn; Test 2 chỉ lấy mẫu sau khi toàn Test 1 hoàn tất. Soft budget mặc định 6 giờ **mỗi lần gọi run**, code75 = partial/resumable, code khác = lỗi cần xử lý. Thời gian chuẩn bị tải model/preflight nằm ngoài budget này. Chưa khẳng định toàn pilot sẽ xong trong một phiên.

Resume cùng thư mục: chạy lại cell tương ứng hoặc `run`. Resume phiên mới: tải ZIP hoặc Save Version outputs, đưa ZIP vào private Kaggle Dataset rồi attach; bản một cell tự chọn checkpoint đúng mode/config/commit. Nếu có nhiều checkpoint phù hợp, giữ một bản mong muốn. Không tự trộn partial runs. Identity khác precision, code commit, packages, GPU, tokenization hay config sẽ bị từ chối; dùng môi trường cũ hoặc output mới. Preflight giữ precision đã chốt của run đang có.

## Source

- `core.py`: strict Top-1, một MASK mỗi bước, absolute physical32/active8, hard KEEP.
- `runtime.py`: target/AR greedy, shifted feedback, reviewed Fast-dLLM cache provider, real GPU preflight.
- `experiment.py`: paired states, balanced sampling, checkpoint/ZIP.
- `repair.py` / `plans.py`: ma trận source×redrafter, frozen content-aware anchors, offline controls và reverify.
- `selection.py`, `metrics.py`: lựa chọn KEEP và định nghĩa metric.
- `assets.py`, `store.py`: immutable assets, chọn prompt, resume identity và transactional storage.
- `analysis.py`: CSV/JSON/Markdown, paired deltas, prompt-cluster bootstrap và 10 plots.
- `tests/`: scheduler, indexing, selectors, pipeline fixture, SQLite recovery, tiny random official-model audits.
- `scripts/build_notebook.py`: nguồn chung để tạo notebook và full copyable cells.

## Kiểm chứng

Xem [docs/VALIDATION.md](docs/VALIDATION.md) cho kết quả local thực tế. Kiểm thử tiny random CPU chỉ kiểm logic/caches; fixture không thay thế pretrained GPU checks. **Chưa chạy Kaggle T4 hay thí nghiệm 100+100 trong phiên phát triển này.** Preflight trên model thật là bắt buộc và dừng khi cached/full hoặc target reference/verification không khớp.

Test 1 ghi riêng old-context verifier agreement, true-reference positional matches và direct rejoin. Test 2 so với fresh-L cùng physical layout và horizon L=32-r (r 1-based). Mỗi candidate sửa xong được verifier chấm lại và kiểm longest accepted prefix/correction khớp reference. KEEP vẫn là proposal chưa được commit online. Mọi latency là cold replay có đồng bộ GPU, không gán cho online rollback hoặc end-to-end speedup.
