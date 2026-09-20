# Edge-Optimized IoT Intrusion Detection

Compressing a large, accurate intrusion-detection neural network small enough to run on
router-class hardware — and measuring honestly what each compression stage actually buys.

The contribution is not the detection. A shallow MLP already separates benign from
malicious network flows fairly well. The contribution is the **measured trade-off curve**:
how far the model can be shrunk, by which technique, before detection quality degrades.

## Why on-device

Cloud-based network monitoring requires a router to stream local traffic telemetry to a
third-party server. That exposes private network metadata continuously, and the round trip
adds latency during which a fast-propagating botnet spreads further. Local inference
removes both problems, at the cost of a hard constraint: the model must fit in kilobytes.

## The pipeline

```
IDS2018 flow CSVs →  clean / de-leak / scale  →  ┌─ TEACHER  [512→256→128→64]
                                                 │      │ soft logits (T=4)
                                                 │      ▼
                                                 ├─ STUDENT-KD    [32→16]
                                                 └─ STUDENT-PLAIN [32→16]   ← ablation
                                                        │
                                          structured prune → rebuild → fine-tune
                                                        │
                                                 static PTQ → INT8
```

**Five models are benchmarked, not four.** `student-plain` is trained on hard labels only,
under settings identical to `student-kd` in every respect except the distillation term.
Without it, "the student reached 0.98 F1" is uninterpretable — a small MLP might reach
0.98 unaided. The gap between those two rows is the entire empirical claim of the
distillation stage.

## Setup

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/Scripts/python.exe torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/Scripts/python.exe numpy pandas scikit-learn matplotlib pyyaml joblib tqdm pytest requests
```

## Running it

Smoke test on generated data — about a minute, proves the wiring works:

```bash
.venv/Scripts/python.exe scripts/run_pipeline.py --synthetic
```

Quick run on 5% of the real flows:

```bash
.venv/Scripts/python.exe scripts/run_pipeline.py --sample-frac 0.05
```

Full run:

```bash
.venv/Scripts/python.exe scripts/run_pipeline.py
```

Tests:

```bash
.venv/Scripts/python.exe -m pytest -v
```

### Getting the dataset

The pipeline uses **CSE-CIC-IDS2018** and downloads it automatically on first run. It
defaults to `Friday-02-03-2018_TrafficForML_CICFlowMeter.csv` (336 MB), which is the
**botnet capture day** — it carries `BENIGN` and `Bot` labels, giving exactly the binary
botnet-detection problem this project targets.

To use other attack families, add filenames to `data.files` in `configs/default.yaml`.
The capture days and their attacks are listed in `IDS2018_DAYS` in
[`src/edge_ids/data/download.py`](src/edge_ids/data/download.py).

**Why not CIC-IDS2017?** The project originally targeted it, but the CIC host now answers
every dataset path with an HTML download form rather than the archive — verified across
four candidate URLs, all returning the same 108,784-byte page. IDS2018's processed CSVs
sit in a public S3 bucket and fetch directly, so it is the reproducible choice. The 2017
loader is still present (`source: ids2017`) if you obtain the archive by hand; both
datasets are CICFlowMeter output with the same feature style, and the preprocessing
handles either.

## Results

Written to `artifacts/results/`: `metrics.csv`, `comparison.md`, and plots for confusion
matrices, ROC/size trade-off, and latency.

### How to read them

**F1 and FPR, not accuracy.** Benign flows outnumber attacks roughly four to one, so a
model that predicts "benign" for everything scores ~0.80 accuracy and detects nothing.
False-positive rate matters especially: this model's output gates a firewall block, so a
false positive means a legitimate device — a thermostat, a camera, someone's laptop —
gets cut off the network.

**Two size columns.** `size_kb` is measured bytes on disk after `torch.save`.
`weight_memory_kb` is the parameters alone at their true precision (4 bytes FP32, 1 byte
INT8). They diverge sharply on small models, and both are reported because reporting only
the flattering one would misstate the result.

### Findings established during development

These were measured, not assumed, and several run against the intuition the project
started with:

1. **INT8 does not shrink a model this small.** Per-channel scales, zero-points and
   container metadata are fixed overhead. Above ~10K parameters they are negligible
   against the 4× weight saving (a teacher-scale model compresses 819.8 KB → 231.9 KB,
   −72%). Below a few thousand they dominate, and the quantized *file* is larger than the
   float one — while the weight memory still drops 4×, exactly as theory predicts.

2. **INT8 is substantially slower here, not faster.** On the smoke-test run the quantized
   student measured ~549 µs against ~71 µs for the pruned FP32 model. For a network this
   small, per-inference quantize/dequantize overhead swamps the gain from integer
   arithmetic. The benefit of INT8 is weight memory and integer-only target hardware —
   not wall-clock speed on x86.

3. **Distillation and pruning deliver essentially all of the compression.** Teacher to
   pruned student is already a ~99% reduction before quantization is applied at all.

4. **`torch.nn.utils.prune` would not have worked.** It masks weights rather than removing
   them — the zeroed values still occupy full FP32 tensors, so parameter count and file
   size are unchanged. This project rebuilds layers physically, copying surviving neurons
   into smaller `nn.Linear` modules, which is why the reported KB figures move at all.

## Layout

```
configs/default.yaml          every hyperparameter, one file
src/edge_ids/
  data/       download.py  preprocess.py  dataset.py
  models/     teacher.py   student.py
  training/   loops.py     train_teacher.py  distill.py
  compression/prune.py     quantize.py
  evaluation/ metrics.py   benchmark.py      report.py
scripts/run_pipeline.py       one command, end to end
tests/                        104 tests
docs/superpowers/             design spec and implementation plan
```

## Implementation notes

- **Leakage.** `Flow ID`, source/destination IP, ports and `Timestamp` are dropped before
  training. Left in, the model memorises attacker addresses and reports ~99.99% accuracy
  having learned nothing that generalises.
- **Scaler fitted on the training split only.** Fitting before splitting leaks the test
  distribution into training.
- **Distillation uses the `T²` correction.** Soft-target gradients scale as `1/T²`;
  without the factor, raising the temperature silently turns distillation back into plain
  cross-entropy. Three tests pin this.
- **The student has no BatchNorm.** It would need fusion before static quantization and
  its running statistics would have to be sliced in step with the pruning rebuild.
- **The quantization backend is auto-detected.** This PyTorch build ships only `onednn`;
  a hardcoded `fbgemm` fails at conversion time, after the entire pipeline has run.
- **Latency is measured on one thread**, warmed up, reported as median and p95.
