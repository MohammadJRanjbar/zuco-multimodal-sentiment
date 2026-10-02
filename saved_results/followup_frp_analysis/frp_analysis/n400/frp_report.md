# Fixation-related potentials (ZuCo)

Fixations located in the sentence EEG: 95.2% on average (range 94.0%-96.7%) over 9 readers; 38,514 word epochs. Segment length vs ZuCo first-fixation duration: r = 1.000 (should be close to 1).

Readers without per-fixation EEG in their files (not usable): ZDN, ZGW, ZJM (3 of 12).

## Timing check

Occipital peak at 102 ms (2.14 microvolts; baseline SD 0.14). **Pass:** a clear lambda/P1 response at the expected latency, so onsets are aligned.
Centro-parietal mean amplitude 300-500 ms: -0.06 microvolts. Plot: `plots/frp_grand_average.png`.

## N400 regression (trial level, reader fixed effects)

38,354 reader-word observations from 400 sentences. Coefficients are microvolts per standard deviation of the predictor; 95% CIs from a sentence bootstrap.

| predictor | beta | 95% CI |
|---|---:|---|
| surprisal | -0.017 * | [-0.037, -0.001] |
| zipf | -0.020 | [-0.046, 0.005] |
| length | -0.006 | [-0.022, 0.010] |
| relative_position | 0.035 * | [0.023, 0.046] |
| is_content | -0.022 * | [-0.039, -0.002] |
| log_first_fix | -0.006 | [-0.033, 0.019] |
| log_next_fix | 0.061 * | [0.038, 0.086] |
| in_lexicon | 0.008 | [-0.021, 0.036] |
| valence | -0.007 | [-0.020, 0.004] |
| abs_valence | -0.001 | [-0.028, 0.025] |

\* CI excludes 0.

* **Positive control (surprisal):** more surprising words give a more negative N400, as expected; the pipeline detects a known semantic effect.
* **Sentiment terms (valence, |valence|):** no effect beyond the lexical and timing controls.
