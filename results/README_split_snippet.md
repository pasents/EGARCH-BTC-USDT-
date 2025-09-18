### Train / Validation / Test Protocol

- **Frequency:** raw
- **Train:** 2018-02-01 → 2021-12-31  (1427 (52.9%))
- **Validation:** 2022-01-06 → 2023-12-31  (725 (26.9%))
- **Test (Out-of-Sample):** 2024-01-06 → 2025-07-06  (547 (20.3%))
- **Embargo:** 5 bar(s) at each boundary to reduce information leakage due to serial dependence.

**Why these periods?**
- **2019–2020 (Train):** mixed/sideways markets with a gradual uptrend give EGARCH exposure to low–mid volatility regimes.
- **2021–2022 (Validation):** extreme bull (2021) followed by a deep bear (2022) stress-test the hyperparameters (take-profit %, stop k·σ, refit cadence) across opposite regimes.
- **2023–present (Test):** recovery and new ATHs with more stepwise advances. We freeze hyperparameters after validation and evaluate **once** on this unseen period.

**Protocol details**
- Hyperparameters are chosen **only** on the validation set (robustness plateaus preferred over single maxima).
- Model parameters (EGARCH coefficients) are **re-estimated walk-forward** using only past data at each step.
- The test set is untouched until the end and is evaluated in a single pass.