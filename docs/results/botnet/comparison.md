| model | F1 | accuracy | recall | FPR | ROC-AUC | params | size (KB) | vs teacher | weights (KB) | vs teacher | median (us) | p95 (us) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| teacher | 0.9956 | 0.9984 | 0.9969 | 0.0013 | 1.0000 | 209,346 | 834.20 | 0.0% | 817.76 | 0.0% | 401.0 | 535.7 |
| student-plain | 0.9973 | 0.9990 | 0.9961 | 0.0003 | 1.0000 | 2,738 | 13.78 | 98.3% | 10.70 | 98.7% | 59.6 | 105.7 |
| student-kd | 0.9965 | 0.9987 | 0.9978 | 0.0011 | 1.0000 | 2,738 | 13.74 | 98.4% | 10.70 | 98.7% | 59.0 | 111.8 |
| student-pruned | 0.9963 | 0.9986 | 0.9973 | 0.0011 | 1.0000 | 1,242 | 7.92 | 99.1% | 4.85 | 99.4% | 58.3 | 107.9 |
| student-pruned-int8-static | 0.9956 | 0.9983 | 0.9947 | 0.0008 | 1.0000 | 1,242 | 9.08 | 98.9% | 1.21 | 99.9% | 591.8 | 754.7 |
| student-pruned-int8-dynamic | 0.9940 | 0.9978 | 0.9921 | 0.0009 | 1.0000 | 1,242 | 6.66 | 99.2% | 1.21 | 99.9% | 614.4 | 766.9 |

`size (KB)` is the measured file on disk. `weights (KB)` is the parameters
alone at their true precision. They diverge on small models because
quantization metadata is fixed overhead - see spec section 9.
