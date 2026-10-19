# Suffix Reusability & Selective Redrafting

Vòng đầu: **100 MATH-500 + 100 GSM8K**, seed 42. Tối đa 5 shared prefixes/prompt, 32 candidate tokens, tối đa **2.000 Test 1 proposals**. Test 2 lấy 200 rejection events có suffix không rỗng, phân tầng benchmark/source/rejection bin và ưu tiên dLLM theo tỷ lệ 2:1 khi đủ mẫu. HumanEval và general instruction chưa chạy trong vòng này.

Đây là hai thí nghiệm **offline mechanism/headroom**, chưa phải online speculative decoding, chưa chứng minh speedup. Không fine-tune weights. Xem protocol và giới hạn ở [docs/PROTOCOL.md](docs/PROTOCOL.md).

| Vai trò | Checkpoint | Thiết bị / precision |
|---|---|---|
| Verifier/reference | Qwen/Qwen2.5-7B-Instruct | GPU0 INT8; có thể NF4 trước khi bắt đầu run |
| AR drafter | Qwen/Qwen2.5-1.5B-Instruct | GPU1 FP16 |
| Diffusion drafter/infiller | Efficient-Large-Model/Fast_dLLM_v2_1.5B | GPU1 FP16, load lần lượt với AR |

`source-lock.json` khóa model/dataset revisions, official Git revision và hash code remote. `prepare` xác minh hash trước inference. Không dùng CPU/disk offload.

## Chạy trên Kaggle

1. Notebook Settings: **Internet ON, GPU T4 x2**.
2. Repo public: không cần GitHub token. Nếu repo vẫn private: add/enable Kaggle Secret **GITHUB_TOKEN** có quyền đọc repo. Không dán token vào cell. HF_TOKEN tùy chọn.
3. Import bản **một code cell** phù hợp: [smoke5+5](notebooks/math_gsm8k_smoke.ipynb) hoặc [pilot100+100](notebooks/math_gsm8k_pilot.ipynb). Có thể copy từng cell từ [KAGGLE_TWO_CELLS.md](KAGGLE_TWO_CELLS.md). Mỗi cell tự setup và chạy correctness checks; không cần biến `lab`, không cần smoke status trong phiên test thật.
4. Để Save Version chạy thật, dùng notebook pilot riêng chỉ chứa một code cell, chọn **Save Version → Save & Run All**. Không để cell smoke trong notebook này, vì Save & Run All sẽ chạy lại mọi cell từ đầu. Internet vàT4x2 cần bật trong notebook được lưu.
5. Cả hai bản tự xuất ZIP/report/10plots, phân biệt complete/partial/failed và khôi phục ZIP đúng mode/config/commit được attach vào `/kaggle/input`. Kết quả cũ khác code revision được giữ nguyên; launcher chọn output riêng cho revision mới. Phiên Save Version không dùng biến/cache RAM của phiên interactive.
Notebook7cell cũ vẫn được giữ để xem chi tiết từng giai đoạn. Bản một cell dùng runtime/source commit cố định `c4a25fafcdd66884c33ef7251f14a6bc6874a388`, kiểm SHA256 launcher trước khi thực thi; không lấy code mới tùy theo branch head khi Save Version chạy sau này.

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
- `experiment.py`: paired states, isolated repair arms, non-oracle/oracle separation, stratification, checkpoint/ZIP.
- `selection.py`, `metrics.py`: lựa chọn KEEP và định nghĩa metric.
- `assets.py`, `store.py`: immutable assets, chọn prompt, resume identity và transactional storage.
- `analysis.py`: CSV/JSON/Markdown, paired deltas, prompt-cluster bootstrap và 10 plots.
- `tests/`: scheduler, indexing, selectors, pipeline fixture, SQLite recovery, tiny random official-model audits.
- `scripts/build_notebook.py`: nguồn chung để tạo notebook và full copyable cells.

## Kiểm chứng

Xem [docs/VALIDATION.md](docs/VALIDATION.md) cho kết quả local thực tế. Kiểm thử tiny random CPU chỉ kiểm logic/caches; fixture không thay thế pretrained GPU checks. **Chưa chạy Kaggle T4 hay thí nghiệm 100+100 trong phiên phát triển này.** Preflight trên model thật là bắt buộc và dừng khi cached/full hoặc target reference/verification không khớp.

Test 1 ghi riêng old-context verifier agreement, true-reference positional matches và direct rejoin. Test 2 so với fresh-L cùng physical layout và horizon L=32-r (r 1-based). Candidate KEEP chưa được accept; chỉ tính longest correct prefix sau khi so với reference. Mọi latency là cold replay có đồng bộ GPU, không gán cho online rollback hoặc end-to-end speedup.
