fit set n=200, eval set n=200 (disjoint). Fitted on the fit set only: T=1.460, threshold for >=95% recall 0.197, for >=98% recall 0.131.

| Setting | Set | Metrics |
|---|---|---|
| baseline T=1, thr 0.5 | fit | acc 90.5%  recall 93.0% (93/100)  spec 88.0% (88/100)  NLL 0.289  Brier 0.0828  ECE 0.0608 |
| baseline T=1, thr 0.5 | eval | acc 92.5%  recall 95.0% (95/100)  spec 90.0% (90/100)  NLL 0.252  Brier 0.0673  ECE 0.0481 |
| T=1.46, thr 0.5 | fit | acc 90.5%  recall 93.0% (93/100)  spec 88.0% (88/100)  NLL 0.260  Brier 0.0767  ECE 0.0511 |
| T=1.46, thr 0.5 | eval | acc 92.5%  recall 95.0% (95/100)  spec 90.0% (90/100)  NLL 0.230  Brier 0.0636  ECE 0.0489 |
| T=1.46, thr 0.20 | fit | acc 91.0%  recall 95.0% (95/100)  spec 87.0% (87/100)  NLL 0.260  Brier 0.0767  ECE 0.0588 |
| T=1.46, thr 0.20 | eval | acc 92.5%  recall 97.0% (97/100)  spec 88.0% (88/100)  NLL 0.230  Brier 0.0636  ECE 0.0510 |
| T=1.46, thr 0.13 | fit | acc 87.0%  recall 98.0% (98/100)  spec 76.0% (76/100)  NLL 0.260  Brier 0.0767  ECE 0.0587 |
| T=1.46, thr 0.13 | eval | acc 91.0%  recall 98.0% (98/100)  spec 84.0% (84/100)  NLL 0.230  Brier 0.0636  ECE 0.0510 |

Reading: T rescales confidence only (accuracy is unchanged at threshold 0.5); the threshold trades specificity for recall on destructive commands. Values come from a fit set of this size; treat them as indicative, not as a shipped setting. The server still answers at T=1 like the official one.
