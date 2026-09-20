| model | F1 | accuracy | recall | FPR | ROC-AUC | params | size (KB) | vs teacher | weights (KB) | vs teacher | median (us) | p95 (us) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| teacher | 0.3923 | 0.5401 | 0.6422 | 0.4907 | 0.6226 | 210,370 | 838.20 | 0.0% | 821.76 | 0.0% | 235.7 | 420.3 |
| student-plain | 0.3945 | 0.4867 | 0.7234 | 0.5845 | 0.6215 | 2,802 | 14.03 | 98.3% | 10.95 | 98.7% | 60.9 | 110.0 |
| student-kd | 0.3921 | 0.5005 | 0.6969 | 0.5585 | 0.6120 | 2,802 | 13.99 | 98.3% | 10.95 | 98.7% | 60.2 | 87.8 |
| student-pruned | 0.3871 | 0.5470 | 0.6188 | 0.4745 | 0.6212 | 1,274 | 8.04 | 99.0% | 4.98 | 99.4% | 59.0 | 100.0 |
| student-pruned-int8-static | 0.3754 | 0.5849 | 0.5397 | 0.4014 | 0.6208 | 1,274 | 9.08 | 98.9% | 1.24 | 99.8% | 354.1 | 526.8 |
| student-pruned-int8-dynamic | 0.3903 | 0.5286 | 0.6529 | 0.5088 | 0.6207 | 1,274 | 6.66 | 99.2% | 1.24 | 99.8% | 494.5 | 731.9 |

`size (KB)` is the measured file on disk. `weights (KB)` is the parameters
alone at their true precision. They diverge on small models because
quantization metadata is fixed overhead - see spec section 9.
