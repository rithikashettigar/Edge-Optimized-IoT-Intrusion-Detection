# Edge-Optimized Intrusion Detection via Knowledge Distillation and Model Compression

**Status:** Approved for implementation
**Date:** 2026-09-10
**Project root:** `D:\rithika`

---

## 1. Purpose

Build a deep-learning pipeline that takes a large, accurate neural network for network
intrusion detection and compresses it small enough to run on router-class hardware —
without losing the accuracy that made it worth deploying.

The contribution is not the detection itself. A shallow MLP can already separate benign
from malicious network flows reasonably well. The contribution is the **measured
trade-off curve**: how much can this model be shrunk, by which technique, before accuracy
degrades — and how much of the large model's knowledge survives the trip.

### Why on-device rather than cloud

Cloud-based network monitoring requires a router to stream local traffic telemetry to a
third-party server. This has two costs: private network metadata leaves the premises
continuously, and the round trip adds latency during which a fast-propagating botnet can
spread further. Local inference eliminates both. The constraint it imposes — the model
must fit in a few kilobytes and infer in under a millisecond — is what this project
addresses.

---

## 2. Success criteria

The project succeeds if it produces an honest, reproducible comparison table across five
models, populated by measurement rather than assertion.

| Criterion | Target |
|---|---|
| Distilled student F1 | Within 2 percentage points of the teacher |
| Distilled student beats plain student | Measurable F1 gap attributable to distillation |
| Final compressed size | ≥ 95% reduction from teacher, verified by on-disk bytes |
| Inference latency | Reported honestly, single-threaded, warmed up |
| Reproducibility | One command reruns the entire pipeline from raw CSVs |
| Correctness | Pruning rebuild verified equivalent; no data leakage |

**Explicit non-goal:** we do not claim the INT8 model will be dramatically faster on an
x86 laptop. See §9.

---

## 3. Dataset

**CIC-IDS2017**, `MachineLearningCSV` distribution — eight labelled CSVs of pre-extracted
flow features (~2.8M flows), covering benign traffic plus DoS, DDoS, PortScan, brute
force, web attacks, infiltration, and a dedicated **Bot** (botnet) capture.

### Known defects that must be handled

CIC-IDS2017 is widely used and widely mishandled. Each of the following will break or
silently corrupt a naive `pd.read_csv` pipeline:

1. **Leading spaces in column names** — the header is `' Flow Duration'`, not
   `'Flow Duration'`. Every column name gets `.strip()` applied.
2. **Non-UTF8 bytes in labels** — the label `Web Attack – Brute Force` uses a non-ASCII
   dash that raises `UnicodeDecodeError` under default encoding. Read with `latin1`.
3. **Infinities** — `Flow Bytes/s` and `Flow Packets/s` contain `inf` from division by
   zero-duration flows. Replaced with `NaN`, then handled with the NaN policy below.
4. **NaN values** — dropped rowwise (they are a small fraction; imputing invented flow
   statistics is worse than discarding).
5. **Duplicate rows** — present in quantity; dropped before splitting, so the same flow
   cannot appear in both train and test.
6. **Zero-variance columns** — several flag counters are constant across the entire
   dataset. They carry no information and destabilise `StandardScaler`. Dropped.
7. **Severe class imbalance** — roughly 80% benign. Accuracy is therefore an unusable
   headline metric.

### Leakage prevention

Any column that identifies *which capture a row came from* rather than *what the traffic
looked like* is dropped before training. This includes `Flow ID`, `Source IP`,
`Destination IP`, `Source Port`, `Timestamp`, and `Destination Port`.

Left in, these let the model memorise attacker addresses and report ~99.99% accuracy
while having learned nothing generalisable. The drop list is configurable and applied
defensively — columns are dropped if present, so the loader works with either the
`MachineLearningCSV` or `GeneratedLabelledFlows` distribution.

`Destination Port` is a judgement call: it is genuinely informative to a real IDS, but a
detector keyed on port numbers is trivially evaded. It is dropped by default and the
choice is recorded in the config so the alternative can be reported.

### Label and splits

Binary: `BENIGN → 0`, everything else → `1`.

Stratified 60/20/20 train/validation/test. Validation drives early stopping and pruning
fine-tune decisions; test is touched exactly once, at final benchmark time.

### Scaling

`StandardScaler` **fit on the training split only**, then applied to validation and test.
Fitting on the full dataset before splitting leaks test-set distribution into training and
is a common silent error.

Scaled features are clipped to `[-5, 5]`. Network flow features are heavy-tailed; a
handful of extreme outliers otherwise dominate the quantization calibration range and
destroy INT8 resolution for the bulk of the data.

The fitted scaler is persisted to disk. Inference must apply byte-identical preprocessing
or the deployed model is not the model that was evaluated.

---

## 4. Models

### 4.1 Teacher — `src/edge_ids/models/teacher.py`

`TeacherMLP`: `[n_features → 512 → 256 → 128 → 64 → 2]`

Each hidden block is `Linear → BatchNorm1d → ReLU → Dropout(0.3)`.

Deliberately over-parameterized — roughly 211K parameters (~824 KB in FP32) at ~70 input
features. It is not meant to be deployable. It is meant to be as accurate as this data
permits, establishing the ceiling everything else is measured against.

Trained with standard cross-entropy, class-weighted to counter imbalance.

Exact parameter count depends on how many features survive the drops in §3; all sizes in
this document are reported as measured at run time, not carried forward from here.

### 4.2 Student — `src/edge_ids/models/student.py`

`StudentMLP`: `[n_features → 32 → 16 → 2]`

Plain `Linear → ReLU` stack. Roughly 2.8K parameters (~11 KB FP32).

**No BatchNorm, by design.** BatchNorm requires layer fusion before static quantization
and complicates the structured-pruning rebuild, because the running statistics must be
sliced in step with the surviving neurons. Keeping the student a bare stack of Linear
layers makes both compression stages clean and auditable. The accuracy cost at this scale
is negligible; the reduction in ways-to-be-subtly-wrong is not.

The class exposes optional `QuantStub`/`DeQuantStub` wrapping, enabled for the static PTQ
path and inert otherwise.

### 4.3 The five benchmarked models

| # | Model | Purpose |
|---|---|---|
| 1 | Teacher | Accuracy ceiling |
| 2 | Student-plain | **Ablation** — same architecture, trained on hard labels only |
| 3 | Student-KD | Distilled from the teacher |
| 4 | Student-pruned | Model 3, structurally pruned and fine-tuned |
| 5 | Student-INT8 | Model 4, post-training quantized |

Model 2 is what makes the results meaningful. Without it, "the student reached 98% F1" is
uninterpretable — a small MLP might reach 98% on this data unaided. The gap between
models 2 and 3 is the entire empirical claim of the distillation stage, and it is the
number an examiner should ask for.

---

## 5. Knowledge distillation

```
L = α · T² · KL(student_soft ‖ teacher_soft) + (1 − α) · CE(student_logits, y)
```

where `student_soft = log_softmax(student_logits / T)` and
`teacher_soft = softmax(teacher_logits / T)`.

Defaults: `T = 4`, `α = 0.7`.

The teacher runs in `eval()` mode under `torch.no_grad()`; its logits are targets, not a
path for gradients.

The hard-label `CE` term uses the same class weights as the teacher's training, so the
plain-student ablation (§4.3) differs from the distilled student in exactly one respect —
the presence of the KD term — and the comparison stays honest.

**The `T²` factor is mandatory.** Soft-target gradients scale as `1/T²`. Omitting the
correction makes the KD term shrink quadratically as temperature rises, so raising `T`
silently converts the run into plain cross-entropy training while appearing to distill.
This is asserted in a unit test rather than trusted.

---

## 6. Structured pruning

Neurons in each hidden layer are ranked by the L2 norm of their weight vectors. The
weakest fraction is removed.

**Default schedule:** a 50% total reduction of hidden neurons, reached in three
iterations of ~20% each, with 5 fine-tuning epochs between iterations. The output layer
is never pruned. A floor of 4 neurons per hidden layer prevents a layer collapsing to
nothing at aggressive ratios.

**Removal is physical, not masked.** `torch.nn.utils.prune` applies a binary mask: the
zeroed weights still occupy full FP32 tensors, so parameter count and file size are
unchanged. A pipeline that reports "50% pruned" on masked weights and then reports an
unchanged file size — or worse, reports a computed rather than measured size — is
reporting a result it did not obtain.

Instead, layers are rebuilt: a new `nn.Linear` is constructed with fewer `out_features`,
surviving rows are copied from the old weight matrix, and the *following* layer's
`in_features` is reduced by copying the matching surviving columns. The result is a
genuinely smaller network.

Applied iteratively — prune a fraction, fine-tune (with the teacher still supervising),
repeat — rather than one-shot, which damages accuracy considerably more at the same final
sparsity.

A unit test asserts that a rebuild at 0% pruning produces outputs numerically identical
to the original model, catching row/column transposition errors in the copy logic.

---

## 7. Quantization

The brief specifies weights **and activations** in INT8, which means **static** post-
training quantization:

- `QuantStub` / `DeQuantStub` at the model boundary
- Backend **auto-detected** from `torch.backends.quantized.supported_engines`, preferring
  `fbgemm`, falling back to `onednn` then `qnnpack`. The installed build
  (torch 2.14.0+cpu) exposes only `onednn`, so a hardcoded `fbgemm` would fail at
  conversion time. The backend actually used is recorded in the results.
- Calibration over 200 **training** batches — never validation or test data
- `torch.ao.quantization.convert` to produce the final INT8 model

`quantize_dynamic` is also run and reported as a second row. It quantizes weights only,
needs no calibration, and is robust. If static PTQ hits an `fbgemm` issue on Windows, the
dynamic result still demonstrates the memory reduction and the pipeline still completes.

---

## 8. Evaluation

### Metrics, per model

Accuracy, precision, recall, **F1**, ROC-AUC, **false-positive rate**, and the full
confusion matrix.

FPR is called out because of what a false positive *means* in this system. The mitigation
layer this model feeds blocks a source IP. A false positive is a legitimate device — a
thermostat, a camera, someone's laptop — being cut off the network. A model with excellent
recall and a poor FPR is not deployable regardless of its headline accuracy.

### Size

Two columns, for the reason given in §9:

- `size_kb` — **actual bytes on disk** after `torch.save`. Computed sizes cannot detect
  the masked-pruning failure described in §6, so this is the authoritative number for
  whether compression really happened.
- `weight_memory_kb` — parameters alone at their true precision (4 bytes FP32, 1 byte
  INT8), isolating the weight saving from serialization overhead.

### Latency

- `torch.set_num_threads(1)` — a router core, not a laptop's full parallelism
- Warmup iterations discarded before timing
- Many repeats; report **median and p95**, not mean (which one outlier distorts)
- Both single-sample (the realistic per-flow case) and batched throughput

### Outputs

`artifacts/results/metrics.csv`, a markdown comparison table, and plots: confusion
matrices, ROC curves, size-vs-F1 trade-off scatter, and latency bars.

---

## 9. Stated expectations and measured findings

**INT8 may not measurably reduce latency on an x86 laptop.** For a network this small,
per-inference quantize/dequantize overhead can offset the gain from integer arithmetic.

This was recorded in advance so that the benchmark result — whatever it turned out to be
— would be reported as a finding rather than explained away afterwards. A project that
predicts its own negative result and measures it anyway is doing science; one that
discovers it and quietly reframes it is not.

### Measured: INT8 does not shrink a model this small

Measured on the installed build, `torch.save` of the state dict:

| model | params | FP32 KB | static INT8 | change |
|---|---|---|---|---|
| teacher-scale (512,256,128,64) | 208,962 | 819.82 | 231.88 | −71.7% |
| wide student (128,64) | 17,474 | 70.96 | 27.23 | −61.6% |
| student (32,16) | 2,834 | 13.77 | 10.23 | −25.7% |
| pruned student (16,8) | 1,290 | 7.77 | 8.29 | **+6.7%** |
| pruned to floor (8,4) | 614 | 5.21 | 7.35 | **+41.1%** |

Per-channel scales, zero-points, and zip-container metadata are **fixed overhead**. Above
roughly ten thousand parameters they are negligible against the 4× weight saving; below a
few thousand they dominate, and the quantized file is larger than the float one.

The mechanism is not broken — static INT8 agrees with the trained FP32 model on >95% of
test predictions. It is an accounting fact about small models.

**Consequences for this project:**

1. Distillation and pruning deliver essentially all of the compression. Teacher at 819.82
   KB to a pruned student at 7.77 KB is a 99.1% reduction before quantization is applied
   at all.
2. The results table therefore reports **two** size columns: `size_kb`, the measured
   file on disk, and `weight_memory_kb`, the parameters alone at their true precision
   (4 bytes FP32, 1 byte INT8). The first is what this pipeline actually produces; the
   second isolates the 4× weight reduction from the container overhead, which a real
   embedded deployment writing a flat weight blob would not carry.
3. Reporting only whichever number flattered the conclusion would be the failure mode
   §9 exists to prevent. Both are reported, with the gap explained.

---

## 10. Repository layout

```
D:\rithika\
├── README.md
├── pyproject.toml                 uv-managed, Python 3.12, CPU-only torch
├── configs/default.yaml           every hyperparameter, one file
├── src/edge_ids/
│   ├── config.py                  load and validate config
│   ├── data/
│   │   ├── download.py            fetch + unzip CIC-IDS2017, manual fallback
│   │   ├── preprocess.py          clean, de-leak, label, scale, split
│   │   └── dataset.py             torch Dataset / DataLoader
│   ├── models/
│   │   ├── teacher.py             TeacherMLP
│   │   └── student.py             StudentMLP
│   ├── training/
│   │   ├── loops.py               shared train/eval epoch, early stopping
│   │   ├── train_teacher.py
│   │   └── distill.py             KD loss, student training, plain baseline
│   ├── compression/
│   │   ├── prune.py               rank, rebuild, iterative fine-tune
│   │   └── quantize.py            static PTQ + dynamic fallback
│   └── evaluation/
│       ├── metrics.py
│       ├── benchmark.py           on-disk size, single-thread latency
│       └── report.py              tables and plots
├── scripts/run_pipeline.py        one command, end to end
├── tests/
├── data/                          raw + processed CSVs (gitignored)
└── artifacts/                     models, scaler, results (gitignored)
```

Each module has one responsibility and a defined interface. Training loops live in
`loops.py` and are shared by the teacher, the plain student, the distilled student, and
the pruning fine-tune, rather than being written four times.

---

## 11. Testing

Test-driven where the logic is subtle enough to be silently wrong:

| Test | Guards against |
|---|---|
| KD loss equals CE when `α=0` | Loss-blending wired backwards |
| KD term scales correctly with `T` | Missing `T²` factor (§5) |
| Prune-rebuild at 0% is numerically identical | Row/column transposition in copy logic |
| Pruning reduces real parameter count | Masking mistaken for removal (§6) |
| Leakage columns absent after preprocessing | Silent reintroduction of identifiers |
| No NaN or inf survive preprocessing | Defect handling regressions (§3) |
| Scaler fitted on train split only | Test-set leakage through normalization |
| Quantized file is smaller than FP32 | Quantization silently not applied |

---

## 12. Execution

`uv`-managed venv on Python 3.12 with CPU-only torch.

`scripts/run_pipeline.py` runs every stage end to end and accepts `--sample-frac` for
smoke-testing on a fraction of the 2.8M flows before committing to a full run. Each stage
is also independently runnable, so a failed quantization step does not require retraining
the teacher.

Expected full-run cost on CPU: teacher training roughly 15–30 minutes, everything
downstream considerably less.

### Risks

| Risk | Mitigation |
|---|---|
| ~224 MB UNB download is slow or unavailable | Manual-download fallback with explicit instructions and path check |
| Static PTQ fails on this backend | Backend auto-detected; dynamic quantization path implemented and reported regardless |
| Full-dataset CPU training too slow | `--sample-frac`, stratified so class balance is preserved |
| Student too small to hold accuracy | Width is configurable; the trade-off gets reported rather than hidden |

---

## 13. Default hyperparameters

All of these live in `configs/default.yaml` and are overridable from the command line.
They are recorded here so a run can be reproduced from the spec alone.

| Group | Parameter | Default |
|---|---|---|
| Data | Split | 60 / 20 / 20 stratified |
| | Feature clip after scaling | `[-5, 5]` |
| | Random seed | 42 |
| Teacher | Hidden layers | `512, 256, 128, 64` |
| | Dropout | 0.3 |
| | Optimizer | Adam, lr 1e-3 |
| | Batch size | 1024 |
| | Max epochs | 40, early stopping patience 5 on val F1 |
| Student | Hidden layers | `32, 16` |
| | Optimizer | Adam, lr 1e-3 |
| | Max epochs | 60, early stopping patience 8 on val F1 |
| Distillation | Temperature `T` | 4.0 |
| | `α` (KD weight) | 0.7 |
| Pruning | Total hidden-neuron reduction | 50% |
| | Iterations | 3 |
| | Fine-tune epochs per iteration | 5 |
| | Minimum neurons per layer | 4 |
| Quantization | Backend | auto (`fbgemm` → `onednn` → `qnnpack`) |
| | Calibration batches | 200 |
| Benchmark | Latency warmup / measured iterations | 100 / 1000 |
| | Threads | 1 |

Both students are trained under identical settings; the only difference is the KD term.
