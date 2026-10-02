# Fixation-related potentials (ZuCo)

Fixations located in the sentence EEG: 94.9% on average (range 94.5%-95.3%) over 2 readers; 8,283 word epochs. Segment length vs ZuCo first-fixation duration: r = 1.000 (should be close to 1).

## Timing check

Occipital peak at 98 ms (1.28 microvolts; baseline SD 0.11). **Pass:** a clear lambda/P1 response at the expected latency, so onsets are aligned.
Centro-parietal mean amplitude 300-500 ms: -0.02 microvolts. Plot: `plots/frp_grand_average.png`.

## N400 regression (trial level, reader fixed effects)

8,259 reader-word observations from 400 sentences. Coefficients are microvolts per standard deviation of the predictor; 95% CIs from a sentence bootstrap.

| predictor | beta | 95% CI |
|---|---:|---|
| surprisal | -0.004 | [-0.037, 0.030] |
| zipf | 0.006 | [-0.039, 0.050] |
| length | 0.004 | [-0.030, 0.036] |
| relative_position | 0.023 * | [0.003, 0.042] |
| is_content | 0.000 | [-0.033, 0.033] |
| log_first_fix | -0.027 | [-0.066, 0.012] |
| log_next_fix | 0.051 * | [0.010, 0.091] |
| in_lexicon | 0.006 | [-0.051, 0.070] |
| valence | -0.007 | [-0.030, 0.017] |
| abs_valence | -0.006 | [-0.067, 0.053] |

\* CI excludes 0.

* **Positive control (surprisal):** no expected negative surprisal effect; treat the sentiment result below with caution.
* **Sentiment terms (valence, |valence|):** no effect beyond the lexical and timing controls.
