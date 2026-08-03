# Fraud Detection: Tabular and Sequential Approaches

Credit-card fraud detection on the Sparkov-style
[`credit-card-transactions`](https://www.kaggle.com/) dataset (~1.29M transactions,
**~0.5789% fraud**), studied under **two deliberately different problem formulations**:

- a **tabular** formulation — each transaction is classified independently from its own
  attributes;
- a **sequential** formulation — the last 20 transactions of a card are read in time order and
  the most recent one is classified.

The two use different data units and different evaluation protocols, so **their metrics are
reported separately and are not directly comparable**. The point of the project is to show
*how the formulation drives the modelling and, especially, the validation design*.

---

## Repository structure

| Notebook | Formulation | Contents |
|---|---|---|
| `eda_and_tabular_baselines.ipynb` | Tabular | Full EDA + KNN, LDA, Random Forest baselines |
| `eda_and_tabular_boosting_ensembles.ipynb` | Tabular | Full EDA + CatBoost, XGBoost, LightGBM, Soft-Voting Ensemble |
| `sequential_lstm.ipynb` | Sequential | LSTM over 20-transaction per-card windows |

The dataset can be downloaded from [Kaggle](https://www.kaggle.com/datasets/priyamchoksi/credit-card-transactions-dataset)) and placed at
`credit_card_transactions.csv` (default Kaggle path
`/kaggle/input/datasets/priyamchoksi/credit-card-transactions-dataset`).

---

## The two formulations

**Tabular (point-wise).** Each row is an independent sample; the model sees one transaction's
features and predicts fraud. Because samples are independent, **stratified k-fold with shuffling
is applied**, and the CV PR-AUC is computed on out-of-fold predictions.

**Sequential (time-series).** For each card, a sliding window of 20 consecutive transactions
(in time order) predicts whether the *last* transaction is fraudulent. Here a shuffled split
**leaks**, so the protocol is a strict **temporal split** — train on the past,
validate/test on the future.

These are two different questions asked of the same data. The repository keeps them in separate
notebooks and never cross-compares their numbers.

---

## Exploratory Data Analysis — key findings

- **Class imbalance:** fraud is ~0.5789% of transactions. Accuracy is not a useful metric here; the project
  uses **PR-AUC** as the primary metric and class weighting throughout.
- **Fraud rate varies across hours and seasons:** aggregated over the whole dataset, some hours of day and
  winter months show a markedly higher fraud rate than others.
- **Geography & behaviour:** fraud rate varies by state, transaction amount, and home-merchant
  distance (computed with the **haversine** formula, the geographically correct measure on latitude and longitude).
- **Data artifact handled:** every `merchant` value is prefixed with `fraud_` by the simulator,
  for fraud and non-fraud rows alike — so the prefix carries **no signal** and is stripped.

---

## Methodology

### Feature engineering (shared)
Time parts (hour, weekday, month, season, …), haversine distance, and card-holder age.
`unix_time` and `trans_year` are used only for ordering/age and are **excluded from model
features**: under a temporal split every test `unix_time` exceeds every train value, so feeding
them in would be a look-ahead leak.

### Encoding — matched to the model family
- **Baselines notebook (KNN, LDA, Random Forest):** one-hot encoding. KNN and LDA are distance/linear models that need it.
  Random Forest shares the same pipeline.
- **Boosting notebook (CatBoost, XGBoost, LightGBM):** **ordinal encoding**. Trees split on thresholds, so integer codes
  are fine and avoid exploding high-cardinality columns like state.

### Imbalance handling
Class weighting from the **training labels only** — `class_weight`/`auto_class_weights` for
CatBoost, `scale_pos_weight = neg/pos` for XGBoost/LightGBM, `class_weight` for Random Forest. 
KNN and LDA are left unweighted: KNN has no class-weight mechanism, and LDA is used with default priors.

### Evaluation protocol
- **Tabular:** Cross validate **PR-AUC** via out-of-fold (`cross_val_predict`) predictions; the decision
  threshold is tuned on those **OOF predictions of the training set**, then frozen and applied to
  the held-out test set.
- **Sequential:** PR-AUC on a temporal split; the threshold is tuned on a temporal **validation
  slice** and frozen for test.

---

## Results

> PR-AUC is the primary (threshold-independent) metric; Precision/Recall/F1 are at the tuned threshold.

### Tabular — baselines 

| Model | CV PR-AUC | Test PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| KNN |0.4821 | 0.4986 | 0.7089 | 0.5030 | 0.5885 |
| LDA | 0.2021 | 0.2065 | 0.3988 | 0.4124 | 0.4055 |
| Random Forest | 0.9125 | 0.9262 | 0.9377 | 0.8121 | 0.8704 |

### Tabular — boosting ensembles 

| Model | CV PR-AUC | Test PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| CatBoost | 0.9387 | 0.9543 | 0.9449 | 0.8688 | 0.9052 |
| XGBoost | 0.9473 | 0.9631 | 0.9645 | 0.8701 | 0.9149 |
| LightGBM | 0.9482 | 0.9615 | 0.9533 | 0.8834 | 0.9170 |
| Soft Voting | 0.9507 | 0.9641 | 0.9503 | 0.8914 | 0.9199 |

### Sequential — LSTM

| Split | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| Validation | 0.7598 | 0.8304 | 0.6418 | 0.7240 |
| Test | 0.7393 | 0.8412 | 0.6163 | 0.7114 |

---

## How to run

Built for a **Kaggle GPU** runtime. Add the dataset to the notebook,
select a GPU accelerator, and run top to bottom.

`USE_GPU = True` in the boosting notebook toggles the device for all three libraries; set it to
`False` to run on CPU.

**Main dependencies:** `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `plotly`,
`tensorflow`, `catboost`, `xgboost`, `lightgbm` (4.6+), `joblib` — all preinstalled on Kaggle.
