# FinSight AI: evaluation report
*Generated 14 Nov 2026 from the saved metrics files. All data is synthetic.*

## 1. Data

- **5,000 payments**, 250 labelled fraud (5.0%). Fraud types: rapid transfer 90, new beneficiary anomaly 80, suspicious collect 80.
- **120 legitimate hard negatives** (large rent-style payments, shopping bursts) that look unusual but are not fraud.
- **5,000 messages** from 65 message families, 29% scam, in 5 language variants.

## 2. Scam-message classifier (TF-IDF + logistic regression)

Flag threshold: 50 on a 0 to 100 scale.

**Headline: test messages come from message families the model never saw in training.**

|  | Precision | Recall | F1 | PR-AUC | False-positive rate |
|---|---|---|---|---|---|
| Rules only | 1.000 | 0.319 | 0.484 | 0.999 | 0.000 |
| Text model | 0.997 | 0.700 | 0.822 | 0.928 | 0.001 |
| Combined | 1.000 | 0.658 | 0.794 | 0.971 | 0.000 |

For comparison, a random split (same families in train and test) is much easier:

|  | Precision | Recall | F1 | PR-AUC | False-positive rate |
|---|---|---|---|---|---|
| Text model, random split | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |

A near-perfect random-split score on templated text mostly shows the task is easy when families repeat. It is not the number to quote.

Text model by language, on unseen families:

| Language | Messages | Precision | Recall |
|---|---|---|---|
| English | 409 | 1.000 | 0.392 |
| Hindi (script) | 265 | 1.000 | 1.000 |
| Hindi (romanised) | 288 | 1.000 | 1.000 |
| Tamil (script) | 125 | 0.963 | 1.000 |
| Tamil (romanised) | 198 | 1.000 | 1.000 |

## 3. Transaction behaviour model (Isolation Forest)

|  | Precision | Recall | F1 | PR-AUC | False-positive rate |
|---|---|---|---|---|---|
| Isolation Forest | 0.760 | 0.760 | 0.760 | 0.828 | 0.013 |
| Rule baseline | 0.296 | 1.000 | 0.457 | 0.000 | 0.125 |

## 4. Risk fusion

Weights fitted on a 60% train split, review threshold chosen on a 20% validation split, results on the untouched 20% test split (1000 payments, 50 fraud).

|  | Precision | Recall | F1 | PR-AUC | False-positive rate |
|---|---|---|---|---|---|
| Behaviour only | 0.649 | 0.740 | 0.692 | 0.628 | 0.021 |
| Message only | 0.778 | 0.280 | 0.412 | 0.369 | 0.004 |
| Network only | 0.885 | 0.920 | 0.902 | 0.952 | 0.006 |
| Fusion | 0.889 | 0.960 | 0.923 | 0.989 | 0.006 |

Fusion weights (log-odds scale): Behaviour +2.64, Message +2.49, Network +5.07.

Share of payments flagged, by kind of payment (test split):

| Kind of payment | Payments | Behaviour only | Message only | Network only | Fusion |
|---|---|---|---|---|---|
| LEGIT BURST | 11 | 36% | 0% | 0% | 0% |
| LEGIT LARGE | 12 | 33% | 0% | 0% | 0% |
| new beneficiary anomaly | 17 | 100% | 24% | 82% | 88% |
| ordinary legit | 927 | 1% | 0% | 1% | 1% |
| rapid transfer | 20 | 90% | 0% | 100% | 100% |
| suspicious collect | 13 | 15% | 77% | 92% | 100% |

Reading this table: the behaviour model misses suspicious collect requests because their amounts look ordinary, while the message and network signals catch them. The behaviour model also raises false alarms on legitimate large payments and shopping bursts, which fusion removes.

## 5. Limitations (please read)

1. **All data is synthetic.** Scores show the pipeline works end to end; they are not real-world accuracy.
2. **The network rule is partly circular.** Mule beneficiaries were generated as very new accounts, and the network indicator looks for exactly that. On real data it would be weaker.
3. **Messages are linked to payments by the generator.** Scam messages were attached to fraudulent payments with fixed probabilities, so the message signal helps fusion more than it would in practice.
4. **Small test split.** About 50 fraud cases, and 13 to 20 per fraud type. Treat per-type figures as noisy.
5. **Message families are parallel across languages.** The same scam scenarios appear in English, Tamil and Hindi, so Tamil and Hindi results benefit from having seen the other languages.
6. **Splits are random across payments, not across accounts.** The same account can appear in train and test; a stricter evaluation would hold out whole accounts.
7. **The 0 to 100 score is a monotone rescaling of model log-odds**, not a calibrated probability of fraud.
8. **The Isolation Forest assumes 5% contamination.** That setting was chosen to match the synthetic fraud rate and would need re-tuning on real data.

## 6. Reproduce

```
python run_pipeline.py
python evaluation/make_report.py
python -m pytest -q
```
