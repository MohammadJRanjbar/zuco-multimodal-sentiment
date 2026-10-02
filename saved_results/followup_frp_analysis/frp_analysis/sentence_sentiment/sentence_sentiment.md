# Sentence sentiment from fixation-locked EEG, with text controls

400 sentences (labels [123, 137, 140]), 9 readers; ridge probes, 5 folds by sentence, 1000 label permutations (200 for single windows). Chance macro-F1 is about 0.33; the permutation null is the fair reference.

| probe | rows | macro-F1 | null mean | null 95th pct | p |
|---|---|---:|---:|---:|---:|
| FRP | reader-averaged | 0.310 | 0.253 | 0.305 | 0.0400 |
| FRP window 0-100 ms | reader-averaged | 0.354 | 0.256 | 0.317 | 0.0199 |
| FRP window 100-200 ms | reader-averaged | 0.236 | 0.258 | 0.318 | 0.7662 |
| FRP window 200-300 ms | reader-averaged | 0.309 | 0.255 | 0.315 | 0.0796 |
| FRP window 300-400 ms | reader-averaged | 0.251 | 0.253 | 0.313 | 0.3881 |
| FRP window 400-500 ms | reader-averaged | 0.253 | 0.250 | 0.304 | 0.3532 |
| FRP window 500-600 ms | reader-averaged | 0.280 | 0.251 | 0.304 | 0.1244 |
| FRP window 600-700 ms | reader-averaged | 0.309 | 0.249 | 0.298 | 0.0398 |
| FRP window N400 300-500 ms | reader-averaged | 0.255 | 0.251 | 0.309 | 0.3184 |
| text form | reader-averaged | 0.392 | 0.252 | 0.313 | 0.0010 |
| text form + text sentiment | reader-averaged | 0.519 | 0.251 | 0.305 | 0.0010 |
| presentation order | reader-averaged | 0.215 | 0.248 | 0.297 | 0.9650 |
| FRP with form and presentation order regressed out | reader-averaged | 0.321 | 0.252 | 0.306 | 0.0320 |
| FRP with form + text sentiment and presentation order regressed out | reader-averaged | 0.302 | 0.252 | 0.306 | 0.0599 |
| FRP | single reader | 0.272 | 0.252 | 0.284 | 0.1209 |

## Reading

* **FRP alone:** above its permutation null (p = 0.0400).
* **Beyond the text:** still above the null after regressing out form and presentation order.
* Text properties alone (rows 'text form...') show how much sentiment the text itself makes predictable from these simple features.
