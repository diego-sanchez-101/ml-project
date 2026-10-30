# Finding the Good Wines

**Author:** Diego Sanchez

A machine learning classification project that asks one question:

> Can a model tell which red wines are **good** (rated 7 or higher by experts), using only their chemical measurements?

The project compares K-Nearest Neighbors and Random Forest, deals with a strong class imbalance (only about 13.5% of the wines are "Good"), and evaluates every model with **recall and precision** instead of accuracy.

---

## Table of contents

1. [Data source](#data-source)
2. [Project structure](#project-structure)
3. [How to run](#how-to-run)
4. [Steps taken](#steps-taken)
5. [Machine learning techniques](#machine-learning-techniques)
6. [Evaluation: why not accuracy](#evaluation-why-not-accuracy)
7. [Results](#results)
8. [Key findings](#key-findings)
9. [Recommendation](#recommendation)
10. [Limitations and next steps](#limitations-and-next-steps)

---

## Data source

- **File:** `WineQT.csv`. It's a version of the Red Wine Quality dataset, with an added `Id` column. *[https://www.kaggle.com/datasets/yasserh/wine-quality-dataset]
- **Size:** 1,143 wines × 13 columns.
- **Missing values:** none.

| Column type | Columns |
|---|---|
| Features (11 chemical measurements) | fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol |
| Original target | `quality`: an expert score, 3–8 in this file |
| Identifier | `Id`: dropped before modeling |

### Distribution of the quality score

| Quality | Wines |
|---|---|
| 3 | 6 |
| 4 | 33 |
| 5 | 483 |
| 6 | 462 |
| 7 | 143 |
| 8 | 16 |

| Statistic | Value |
|---|---|
| Mean | 5.66 |
| Std | 0.81 |
| Median (50%) | 6 |
| 75th percentile | 6 |

Almost every wine scores 5 or 6.

---

## How to run

1. Install the dependencies:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
   ```

2. Place `WineQT.csv` next to the notebook, and update the path in the `pd.read_csv(...)` cell.
3. Open `Project_Wine.ipynb` and run **Kernel → Restart Kernel and Run All Cells**.

All random steps use a fixed `random_state`, so the results are reproducible.

---

## Steps taken

### 1. Data checks
- Confirmed there were no missing values and no repeated `Id`s, so no wine was recorded twice.
- Dropped the `Id` column. It is unique for every row, so it carries no information about quality. Keeping it would also hide duplicate measurements.

### 2. Defining the target
- Created a binary target, `quality_classif`:
  - **Good:** quality ≥ 7 (159 wines, about 14%)
  - **Standard:** quality ≤ 6 (984 wines)
- **Why 7?** The median and the 75th percentile are both 6, so a 6 is a typical wine, not a standout one. A cutoff at 6.5 makes "Good" a genuinely above-typical rating.
- Removed `quality` from the features, since the target is built from it. Keeping it would be **target leakage**: the model could simply read the answer.

### 3. Removing duplicate measurements ("twins")
- Without the `Id`, **125 rows** (10.9%) were exact copies of another row in every measurement and in quality.
- KNN predicts from the most similar wines. A test wine whose twin is in the training set would be classified correctly "for free", inflating the scores.
- After removing them: **1,018 wines**, 137 Good (13.5%) and 881 Standard.

### 4. Train/test split
- 80/20 split with `stratify`, so both parts keep the same share of Good wines:
  - **Train:** 814 wines (110 Good)
  - **Test:** 204 wines (27 Good)
- An earlier split without stratification put only 20 Good wines in the test set (about 10%), which made the evaluation noisy. Stratifying fixed that.

### 5. Modeling and evaluation
See the next two sections.

---

## Machine learning techniques

| Technique | Purpose |
|---|---|
| **K-Nearest Neighbors (KNN)** | Classifies a wine by majority vote of its *k* most similar wines |
| **Feature scaling** (`StandardScaler`) | Puts all features on the same scale, so large-range features don't dominate KNN's distances. Fitted on the training data only. |
| **Distance weighting** (`weights="distance"`) | Closer neighbors get a bigger vote in KNN |
| **Random Forest** | An ensemble of decision trees that vote. It doesn't need scaling. |
| **Balanced class weights** (`class_weight="balanced"`) | Makes mistakes on the rare Good class cost more during training |
| **Random oversampling** | Duplicates Good wines in the training data until the classes are balanced (704 / 704) |
| **SMOTE** | Creates *synthetic* Good wines between real neighboring Good wines (704 / 704, 0 duplicates) |
| **Stratified 5-fold cross-validation** | Tests each model on 5 different folds of the training data for a more reliable estimate than a single split |

**Leakage safeguards:** the scaler, oversampling and SMOTE were applied to **training data only**. The test set was never used to fit anything.

---

## Evaluation: why not accuracy

A "model" that always predicts **Standard** reaches about **86.8% accuracy** on the test set, and finds **zero** Good wines. Accuracy hides exactly the failure that matters here, so every model is judged on the **Good** class:

- **Recall:** of the truly Good wines, how many did the model find?
- **Precision:** of the wines the model labelled Good, how many really were?

Confusion matrices are used throughout to show the actual counts behind these rates.

---

## A note on the numbers 

The models in this project were not all trained and tested on the same data. The data changed twice, so the results come from three versions of the split. **Only results from the same version can be compared directly.** 

### Why the train and test sizes differ 

1. Early experiments | All 1,143 wines, twins included, random split: 
		Train (914) - Test (229) - Good in test: 32 (First three KNN models).
	
2. Twins removed | 1,018 wines, random split:
		Train (814) - Test (204) - Good in test: 20 - No model trained with this. (It had only 20 Good wines, about 10% instead of the real 13.5%, so it was unrepresentative). Replaced by stage 3.
		
3. Final split | 1,018 wines, **stratified** split 
		Train (814) - Test (204) - Good in test: 27 -	Final KNN and all Random Forest models. 

- **Stage 1 → 2:** removing the 125 twins left 1,018 wines. 20% of them (204) went to the test set, and the other 814 to training. 

- **Stage 2 → 3:** the first random split happened to put only 20 Good wines in the test set (about 10%). Re-splitting with `stratify` kept the real share of Good wines in both parts: 27 of 204 (13.2%).

- **Cross-validation** does not use the test set. It splits the 814 training wines into 5 folds, so its confusion matrices add up to 814 wines, 110 of them Good. This is why the early KNN results (for example, precision 0.60) are reported separately. They were measured on a different, easier test set that still contained twins, so they cannot be compared with the Random Forest results.

- ### Which models used scaled data

`StandardScaler` converts each feature into “how far from average is this wine, measured in typical steps. 

-  |  KNN, first model: No - Baseline, before scaling was introduced

-  | KNN, all later versions: Yes - KNN compares wines by distance, so features with large ranges would otherwise dominate.

- | Random Forest (default, class weights, oversampling, SMOTE): No - Trees split on one feature at a time (e.g., "alcohol > 11?"), which works the same in any units 

- | In every case where scaling was used, the scaler was fitted on the training data only and then applied to the test data, so the test set never influenced the preparation. 
- 
	 *Note: SMOTE also uses distances to choose neighbors, so scaling before SMOTE may produce better synthetic wines.


---

## Results

### KNN: early experiments (original data, with duplicates, unstratified)

Test set: 229 wines, 32 Good.

| Model | Good found | False "Good" | Recall | Precision |
|---|---|---|---|---|
| Unscaled, k = 10 | 4 | 6 | 0.13 | 0.40 |
| Scaled, k = 5 | 14 | 14 | 0.44 | 0.50 |
| Scaled + distance weights, k = 5 | 18 | 12 | 0.56 | 0.60 |

*Note: the first model used k = 10, so the first improvement comes from both scaling **and** changing k.*

On the clean data (no duplicates, stratified), the best KNN fell to **recall 0.41 and precision 0.48**, confirming that the twins had inflated its earlier scores.

### Random Forest: single split (same test set, 204 wines, 27 Good)

| Model | Good found | False "Good" | Recall | Precision |
|---|---|---|---|---|
| KNN (scaled + distance) | 11 | 12 | 0.41 | 0.48 |
| **Random Forest (default)** | 14 | **6** | 0.52 | **0.70** |
| Random oversampling | 16 | 11 | 0.59 | 0.59 |
| SMOTE | **19** | 20 | **0.70** | 0.49 |

### Random Forest: 5-fold cross-validation (814 training wines, 110 Good)

| Model | Good found | False "Good" | Mean recall | Mean precision |
|---|---|---|---|---|
| Default | 39 | 27 | 0.355 (std 0.127) | 0.584 |
| Balanced class weights | 65 | 59 | 0.591 (std 0.081) | 0.530 |

*Single-split and cross-validated results are **not** directly comparable. The default forest shows how much they can differ: recall 0.52 vs. 0.36, and precision 0.70 vs. 0.58.*

---

## Key findings

1. **Accuracy is misleading on imbalanced data.** A model can look 87% accurate while never finding a good wine.
2. **Duplicates inflated performance.** Removing the 125 twins lowered KNN's scores, revealing its real performance.
3. **Scaling matters for KNN** and, together with distance weighting, more than quadrupled the number of Good wines found on the original data (4 → 18).
4. **Random Forest outperformed KNN** on the clean data in both recall and precision.
5. **A single test split can be lucky.** Cross-validation gave lower, more realistic scores for the default forest (recall 0.52 → 0.36).
6. **Balancing techniques move along a trade-off; they don't remove it.** Class weights, random oversampling and SMOTE all found more Good wines, but each also produced more false "Good" labels. None improved recall and precision at the same time.

---

## The pick of the tested models

The best model depends on which mistake costs more.

**Scenario chosen: "Good" is a premium label.** Calling an ordinary wine "Good" damages customer trust more than missing a good wine, so **precision** is the priority.

**Pick among the models tested: the default Random Forest.**
- It had the highest precision of every model: **0.70** on the test set and **0.58** cross-validated.
- It made only **6 false "Good" labels** out of 177 Standard wines in the test set.
- **The cost:** it finds only about half of the good wines (recall 0.52 on the test set, 0.36 cross-validated).

If missing good wines were the costlier mistake, SMOTE or balanced class weights would be the better choice.

---

## Limitations and next steps

- **The data is the main limit.** Even the best model is wrong on about 4 in 10 "Good" labels (cross-validated precision 0.58). More data, more Good examples, or additional information about the wines would likely help more than further tuning.

