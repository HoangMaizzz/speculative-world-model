# Test 3 — Oracle-All-Correct Sliding-Window Refill

## Giao thức

- Chỉ Fast-dLLM v2 1.5B draft/redraft -> Qwen2.5-7B-Instruct AR verifier. Fixed FP16, SDPA, target18/10layers trên2GPU, dLLM trênGPU1; không AR drafter, không quantization/offload.
- Original proposal sinh MỚI với B=24 trên canonical pilot prefix. Không cắt first24 của proposal32, không nhập kết quả cũ như kết quả B24.
- Verifier nhận a token đầu rồi correction ở vị trí a (0-based), yêu cầu3<=a<=15. Correction được chốt vào context; suffix cũ S=D24[a+1:], L=23-a. Window kế là1correction +23chưa xác nhận: H=23, E=23-L=a. E dao động3–15, L8–20. Ví dụa7 thìL16,E7; F-L/O-L không luôn dài16.
- Oracle KEEP TẤT CẢ `S[i]==Y_corrected[i]`: giữ cả singleton/noncontiguous islands, không quota25%/2spans, không inject token reference mới. Mọi old-wrong và mọi E mới đều MASK ở Joint.
- Native physical32/block-causal, earliest ABSOLUTE active subblock8, mỗi unmask forward chỉfill1MASK. Không global-H selection, không giả định toàn23token có bidirectional visibility. Prefix/KEEP được khóa trong mọi step bằng shared strict_top1.
- Chỉ một round post-rejection offline; chưa phải full online sliding decoding, không học selector, không báo latency/speedup.

## 6 arms/event, 60 logical arms

| Arm | Horizon chưa xác nhận | Phương thức |
|---|---:|---|
| F-23 | 23 | Fresh joint, toàn MASK |
| O-Joint-23 | 23 | All-correct KEEP, joint redraft+refill |
| O-Sequential-23 | 23 | Dùng O-L xong, khóa ALL L tokens thành prefix rồi extendE |
| F-L | L | Fresh ở horizon cũ |
| O-L | L | All-correct KEEP, không extend |
| KEEP-All-23 | 23 | KEEP toànS kể cả token sai; chỉunmask E |

O-L chạy trước O-Sequential để dùng lại stage1, không sinh lại L ở sequential. Alias chỉ cùng event/horizon/generation-mode/anchor POSITIONS+VALUES; không alias Joint với Sequential dù tình cờ bằng tokens. All-KEEP short có0unmasksteps là hợp lệ. Negative control được phép giữ token sai, Oracle không được.

## Dùng checkpoint và screening

Default unpacked input `/kaggle/input/datasets/ainzkhail/checkpt`: completed recovery pilotfa330e58. Dùng canonical prompt/token IDs và full target reference, không tải datasets hoặc3rdmodel. Input read-only, stream-copy/hash SQLite sang localtemporary như launcher đã sửa.

Reference dưới corrected prefix dùng cùng canonical greedy trajectory của pilot. Với MỖI selected event, kiểm ALL23gold rows bằng actual causal target: teacher-force full gold candidate rồi buộc mọi argmax khớp; disagreement kích actual-model replay từ originalprompt/canonical committed tokens như pilot. Không thay logits bằng gold. Đây là offline reference-audited policy, khôngdeployable verifier.

Mỗi selected original24 và từng unique final generated stage chạy cached/full-prefix equality. Reproduce chính xác selectedB24 và Fresh23 đã dùng khi screening; nếu lệch thì dừng/lưu evidence. Verifier kiểm lại mọi unique final proposal từ corrected prefix với cache độc lập, không dùng old suffix target KV.

### Chọn mẫu có chủ đích, definitions đóng băng

Pool từ tối đa1000DLLM-source pilot contexts, không từ10mẫu mini trước. Seed42, xáo trộn prompt/offset theo seed, round-robin2benchmarks GSM8K/MATH, phủ prompt khác nhau trước offset tiếp của cùngprompt. Mỗi context sinh lại B24; chỉ khi3<=a<=15/reference đủ mới sinh Fresh23 và phân nhóm. Screening dừng ngay khi có nghiệm10distinctprompts,5GSM+5MATH, quotas4/3/2/1. Chưa có Oracle repair outcome nào tại thời điểm chọn/freeze.

Các nhóm mutually exclusive theo thứ tự:

1. **fresh_ceiling (1)**: Fresh23 accept>=22/23.
2. **low_reuse (2)**: khôngceiling và old-correct<=1.
3. **early_rescue (4)**: >=2old-correct, Fresh mismatch f ởoldregion, cóold-correct tại f hoặcf+1.
4. **later_anchor (3)**: >=2old-correct, fởoldregion, khônganchor tại f/f+1, ít nhất80%correctpositions nằm sau f+1; singleton islands vẫn được tính.
5. Các ca khác ghi `other`, không tự lấp quotas.

Nếu không đủ exact quotas, xuất `selection_shortfall` và toàn screening/availability; không ghi complete60arms hoặc tựnớiđiều kiện. Tập10làdiagnostic khôngpopulation; screening có chọn theoFresh outcome để tránhceiling và tìmheadroom, được công bố rõ. Không tuning theo kết quả Oracle.

Screening là GPU work bổ sung ngoài60arms. Các screening drafts chạy cachedstrict1; CHỈselected states được auditfull-prefix và replay; không giả vờ đã có B24 trong checkpoint. Ghi mọi screening trace/calls/reason; checkpoint mỗi state và mỗi arm. Chạy lại cùngcell cùngoutput bỏqua savedstates/arms với cùngruntime/config/input/environment. ClearunusedCUDAcache2GPUs sau state/event, model chỉnạp2trọngsố.

## Metrics và diễn giải

- Primary `A(O-Joint-23)-A(F-23)`, W/T/L, ceilingflags. Joint effect `A(O-Joint)-A(O-Sequential)`. Old repair `A(O-L)-A(F-L)`. A không cộng correction/bonus.
- `A_ext=max(0,A-L)`; positionalcorrect trên H hoặcL, không nhầm aligned matches sau firstmismatch vớiacceptedprefix.
- Với NONKEEP positions: Freshwrong->Oraclecorrect, Freshcorrect->Oraclewrong, bothwrong/bothcorrect, changedIDs; tách REDRAFT/EXTEND.
- Added accepted interval được partition thành: directKEEPcorrected (Freshsai nhưngOracleforceđúng), KEEPvốnFreshđúng, bridgednonKEEPvốnFreshđúng, mutableREDRAFTcorrected, mutableEXTENDcorrected. Không gọi tất cả gained tokens làconditioningbenefit.
- Seq vsF23 đồng thời đổi anchors/schedule nênchỉdescriptive; primaryJointvsFresh làmatchedschedule. Firstmismatchregion là vị trí thất bại, không tựchứng minh causalrootcause.
- Trace ghi absolute predictionrow, block/subblock, carry, futureKEEP nhìnthấy và extensionđã sinh nhìnthấy. Các MASKpad vốn có ởshortcanvas vẫn làcontext; extend khôngtựđộngtạo futureconditioning.

## Artifacts

Default `/kaggle/working/test3_oracle_all/test3_checkpoint.zip`, cósuffixrevision nếu outputcũ tồn tại.

- `events.jsonl`:10events, D24, rejection/correction, sourceprompt/prefix, S/Y, all-correctmask, initialMASKcanvas.
- `arms.jsonl`/`arms.csv`:60logicalarms, proposals/reference, firstmismatch, order/trace/visibility, targetverification, counters/aliases vàattribution.
- `summary.csv`, `events.csv`, `summary.json`, `report.md`:means/W/T/L, all10cases vàgaincauses.
- `cases.html`:all10token-levelviews,3best+3worst theo primarydelta (stableIDties), traceexpandable.
- `screening.jsonl`/`.csv`, `selection.json`:screeningwork/skipreasons/definitions/availability/frozenplan; không gộp như60experimentaltrials.
- `reference_audits.jsonl`:selectednativeB24 cached/fullreplay vàall23correctedtargetrows.
- `pool.json`, `test3-manifest.json`, `runtime.json`, `preflight.json`, `checkpoint.sqlite`, `progress.json`, `execution_status.json` vàfailureevidence.
- Logs riêng `/kaggle/working/test3_setup.log`, `/kaggle/working/test3_run.log`, liveoutput+15sheartbeat.

Chỉ complete với10events và60/60logicalarms. Local mocked/tiny random tests không chứng minhpretrainedGPUquality/headroom. Positiveoracleheadroom cần báo riêngdirectanswerinjection; negativeresult chỉgiới hạn cấu hình/mẫu đãchạy.
