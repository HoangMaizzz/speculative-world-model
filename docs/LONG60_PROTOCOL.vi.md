# Draft60: online speculative decoding và shifted-span diagnostics

Phạm vi:100MATH-500+100GSM8K, seed42, cùng dataset/model revisions trong source-lock.json. Chọn trước inference bằng cùng random.sample như pilot, luân phiên hai benchmark. Hai models: Fast-dLLM-v2-1.5B drafter GPU1; Qwen2.5-7B-Instruct verifier FP16 chia18/10layersGPU0/GPU1. Không load AR1.5B drafter; không INT8/offload. Kaggle T4x2, InternetON.

## Generation thực sự online

- Mỗi vòng sinh mới **60token bằng native strictTop1**, physicalblock32/subblock8. Không đổi physicalblock thành60; không truncate từ B64/B32. Đúng60unmaskforwards/candidate, gồm60MASKupdates. Prefill/cacheupdate được đếm riêng.
- Verifier một causal full-prefix forward trả shiftedrows và một bonusrow. Commit longestmatchingprefix+correction khi reject, hoặc60accepted+bonus khi allaccepted. Rồi draft60mới dưới committedprefix mới. Không có gold/reference hoặc recycledanchors trong đường sinh.
- Output cap256tokens mỗi câu hoặc EOS thực. Round cuối vẫn draft60; chỉ commit còn đủ outputbudget. Báo accepted_draft trên proposal và emitted output riêng.
- DraftEOS: vẫn hoàn tất60MASKtheonativeschedule rồi cắt logicalproposal quaEOS đầu tiên đểverify. Không chấm/recycle token sauEOS. Targetcorrection/bonusEOS dừngngay. Không loại các câu sinh EOSsớm và không thay prompt dựa trên kết quả.
- Cold verifier prefix mỗi round, không targetKVrollback hoặc mutable dLLMblockKVreuse. Đây là online SD có actualtargetverification, nhưng implementation và instrumentation này không dùng để claim serving speedup.

## Kiểm tra numerical correctness

Trước Main: target repeatability/parallel/casual-tail audit60tokens; actualB60dLLM cached/full equality ở physicalphases0/1/7/8/31; mọi native trace check1MASK/forward.

Sau mỗi prompt: target-only canonical greedy generation từ originalprompt, cùngFP16config, tới tối đa376tokens (=256output+120lookahead), dừngEOS. So toàn bộ emittedoutput và acceptedlength từng round với actualcanonicalcontinuation. Reference chỉ được sinh SAU onlinegeneration. Nếu disagreement, fail và giữrawlogits/top1/margins/drafts/ZIP; không sửa onlineoutput bằng gold và không tiếp tục coi prompt là pass. Lookahead đủ cho rejected suffix gần cuốioutputbudget cộng59tokenwindow.

Chưa có pretrained GPU run của release này tại thời điểm triển khai. Local mocked/pure checks không xác nhận numerical/modelquality/GPUruntime.

## Offline matching sau audit

Sau rejection ở chỉ sốa: bỏ token bịreject, old=draft[a+1:], correctedprefix thêmcorrection. Tìm old against canonicaltarget59tokens saucorrection. CorrectionEOS/emptyold giữcount riêng, không recycle. Allacceptedrounds vẫn lưu nhưng không có rejectedsuffix.

Tìm exact contiguous runs>=4tokens ở nonzeroδ=destination-source, radii±8/16/32/59. Giữtoàn bộmaximalruns, longest và chaincùngthứtựsource/destination/khôngoverlap. Chainmaximize tổnglength trong cácmaximalruns đãenumerate; tie fewerblocks/smallerabsδ/coordinates. Không gọi chain này globaloptimaledit alignment. EOS và postEOS bịloại khỏi reusablelabels; censorbiênold/reference lưu rõ.

Mục tiêu: sốevents có>=2/3đoạnrời, độdài/coverage/offset, bridge lengths/content, thayđổioffset. Mọi label là oracleoffline, không detectoronline hoặc implementedretrieval. Rounds trong một câu tươngquan, không xem như independentprompts. Chưa có pairedB24rerun ở cùngcontexts; không kết luận mọi khácbiệt vớiTest4 cũ là causal effect chỉ của B60.

## Artifacts và resume

`/kaggle/working/long60/long60_checkpoint.zip`: SQLitebackup, promptplan/manifests/pins, rawrounds, references, diagnostics, native traces/confidences/order, targettop1+bonus/logitmargins, summaryJSON/CSV, runsCSV, bridgesCSV, casesHTML, tokenpieces, preflight/runtime/status/failureevidence.

SQLitetransaction mỗi round; ZIP mỗi180giây ở roundboundary và mỗi prompt/final. Memorycleanup trên cả2GPU mỗi round. Streaminglog+15sheartbeat; operationalwall gồm peertransfers, setup và audits. Không báo interGPUwall là modelcompute.

Soft6h budget mỗi childsession (notebook pipsetup riêng) checkpoint trước vòng mới; savedrounds resume không chạy lại. Round đang chạy/audit có thể vượt softbudget một đoạn. Rerun cùngcell trong cùngsession tự dùng localcheckpoint. Sessionmới upload ZIP này/unzip và đặtLONG60_RESUME tới folder/ZIP; không dùng checkpointTest1–4 cũ. Revision/config/model/datasetidentity khác sẽ bị từchối. Nếu GPUsystemkill giữa hai ZIP, chỉ bảođảm ZIP gầnnhất; SQLite/WAL trong workingdir có thể córoundsmới hơn khi cònsession.

Runtime phụ thuộcacceptance/EOS, không hứa hoàn tất200câu trong mộtKagglesession. Sau20prompts đầu có thể dùng operationalelapsed/roundcount đểước tínhphầncònlại; độc lập với tốcđộmodelcompute.
