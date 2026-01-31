# Machine Learning Terms & Methods Guide
## RQ2: Land Deals to Litigation Analysis

**Last Updated**: 2026-01-28
**Purpose**: Educational guide explaining all ML/statistical concepts used in the analysis and future possibilities

---

## Table of Contents
1. [Core Statistical Concepts](#1-core-statistical-concepts)
2. [Regression Terminology](#2-regression-terminology)
3. [Model Evaluation Metrics](#3-model-evaluation-metrics)
4. [Data Challenges & Solutions](#4-data-challenges--solutions)
5. [Advanced Techniques Used](#5-advanced-techniques-used)
6. [What Could Be Done With This Data](#6-what-could-be-done-with-this-data)
7. [Recommended Next Steps](#7-recommended-next-steps)

---

## 1. Core Statistical Concepts

### **1.1 Linear Regression**
**What it is**: A method to model the relationship between a dependent variable (outcome) and one or more independent variables (predictors).

**Formula**: `y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε`

**In our context**:
- **Dependent variable (y)**: Number of litigation cases (climate cases or ISDS cases)
- **Independent variables (x)**: Land deal size, number of deals, corruption score, etc.
- **Coefficients (β)**: How much each predictor affects the outcome

**Example**: If β₁ (for Num_Deals) = +0.537, then each additional land deal predicts 0.537 more litigation cases.

---

### **1.2 Ordinary Least Squares (OLS)**
**What it is**: The most common method to estimate linear regression coefficients.

**How it works**: Finds the line/plane that minimizes the sum of squared differences between observed and predicted values.

**Why "least squares"?**:
- Error = (Actual - Predicted)
- Squared error = Error²
- OLS minimizes: Σ(Actual - Predicted)²

**Assumptions**:
1. Linear relationship between predictors and outcome
2. Independence of observations (each country is independent)
3. Homoscedasticity (constant variance of errors)
4. Normality of residuals (errors follow a bell curve)
5. No multicollinearity (predictors aren't too correlated)

**Our violations**:
- ❌ **Non-linear relationship**: Many countries have 0 cases (zero-inflation)
- ❌ **Outliers**: USA has 969 cases vs. median of 0
- ⚠️ **Non-normal residuals**: Heavily right-skewed due to zeros

---

### **1.3 Predictors (Independent Variables)**
**Definition**: Variables used to predict the outcome.

**Our 5 core predictors**:

1. **Total_Deal_Size** (continuous)
   - Total hectares of land acquired in deals
   - Measures: Scale of land acquisition
   - Units: Hectares (ha)

2. **Num_Deals** (count)
   - Number of separate land deals
   - Measures: Frequency of land transactions
   - Units: Count of deals

3. **Corruption_Score** (continuous, 0-100)
   - Corruption Perceptions Index (CPI) from Transparency International
   - Higher = less corrupt, better governance
   - Measures: Quality of governance institutions

4. **Press_Freedom_Score** (continuous, 0-100)
   - From Reporters Without Borders / Freedom House
   - Higher = more press freedom
   - Measures: Civil society strength, ability to mobilize

5. **log_Population** (continuous)
   - Natural logarithm of country population
   - Measures: Country size (logged to reduce skewness)
   - Units: ln(population)

**Why log_Population?**
- Population ranges from ~500,000 to 1.4 billion
- Taking log compresses this range: ln(500k) = 13.1, ln(1.4B) = 21.1
- Makes relationship more linear

---

### **1.4 Dependent Variable (Outcome)**
**Definition**: The variable we're trying to predict.

**Our outcomes**:
- **Model A**: Number of climate litigation cases per country
- **Model B**: Number of ISDS cases per country

**Problem**: Both are **count data** (0, 1, 2, 3...) but we're using continuous linear regression.

**Why this matters**:
- OLS can predict negative values (impossible for counts)
- OLS assumes continuous outcomes (not discrete jumps)
- Zero-inflation (many countries with 0 cases) violates normality assumption

---

## 2. Regression Terminology

### **2.1 Coefficients (β values)**
**What they mean**: The change in the outcome for a 1-unit change in the predictor, holding all other variables constant.

**Example from Model A**:
- **Num_Deals = +0.537**: Each additional land deal predicts 0.537 more climate cases
- **Total_Deal_Size = -0.374**: Each additional unit of deal size predicts 0.374 fewer cases (paradox!)

**Standardized vs. Unstandardized**:
- **Unstandardized** (raw): Uses original units (hectares, number of deals)
- **Standardized** (beta): All predictors scaled to mean=0, SD=1 (allows direct comparison)

**In our analysis**: We use **standardized coefficients** to compare importance across variables with different scales.

---

### **2.2 Intercept (β₀)**
**What it is**: The predicted value when all predictors = 0.

**In our context**: The baseline number of cases for a hypothetical country with:
- Zero land deals
- Zero deal size
- Average corruption
- Average press freedom
- Average population (since it's logged and centered)

**Why it's often not meaningful**: No country actually has all predictors at zero.

---

### **2.3 Residuals (Errors)**
**Definition**: The difference between actual and predicted values.

**Formula**: `Residual = Actual - Predicted`

**Example**:
- USA actual cases: 969
- USA predicted cases: 150
- Residual: 969 - 150 = +819 (huge positive residual = outlier)

**Types of residuals**:
1. **Raw residuals**: Simple difference
2. **Standardized residuals**: Divided by standard deviation
3. **Studentized residuals**: Adjusted for leverage (influence)

**Ideal residual pattern**:
- Randomly scattered around zero
- No patterns when plotted against predictors
- Normally distributed (bell curve)

**Our residual problems**:
- Right-skewed (many negative residuals from countries with 0 cases, few huge positive ones like USA)
- Non-random pattern due to zero-inflation

---

### **2.4 Multicollinearity**
**Definition**: When predictors are highly correlated with each other.

**Why it's bad**:
- Makes coefficients unstable (change dramatically with small data changes)
- Inflates standard errors (makes significance tests unreliable)
- Hard to isolate individual predictor effects

**How to detect**:
- **VIF (Variance Inflation Factor)**: VIF > 10 indicates problematic collinearity
- **Correlation matrix**: r > 0.8 between predictors is concerning

**In our analysis**:
- Total_Deal_Size and Num_Deals are moderately correlated (countries with more deals tend to have larger total size)
- Solution: Keep both because they measure different aspects (scale vs. frequency)

---

## 3. Model Evaluation Metrics

### **3.1 R² (R-squared, Coefficient of Determination)**
**What it is**: Proportion of variance in the outcome explained by the model.

**Range**: 0 to 1 (sometimes reported as 0% to 100%)

**Formula**: `R² = 1 - (SS_residual / SS_total)`

**Interpretation**:
- **R² = 0.00**: Model explains 0% of variance (predictions are just the mean)
- **R² = 0.50**: Model explains 50% of variance
- **R² = 1.00**: Model explains 100% of variance (perfect fit)

**Example from Model A**:
- Train R² = 0.3281 → Model explains 32.81% of variance in training data
- Test R² = 0.1412 → Model explains 14.12% of variance in test data

**Rules of thumb**:
- **R² < 0.10**: Very weak model
- **0.10 < R² < 0.30**: Weak to modest model
- **0.30 < R² < 0.50**: Moderate model
- **R² > 0.50**: Strong model

**Important caveat**: Social science models often have lower R² than physical science models because human behavior is complex and noisy.

---

### **3.2 Test R² vs. Train R²**
**Why split the data?**: To evaluate how well the model **generalizes** to new data.

**Training set**: Data used to fit the model (learn coefficients)
**Test set**: Data held out to evaluate performance

**In our analysis**:
- **80/20 split**: 80% training, 20% testing
- **Model A**: 40 countries for training, 10 for testing
- **Model B**: 90 countries for training, 23 for testing

**What it means**:
- **Train R² >> Test R²**: Overfitting (model memorized training data)
- **Train R² ≈ Test R²**: Good generalization
- **Test R² < 0**: Model predicts worse than just using the mean (complete failure)

**Our results**:
- **Model A**: Train R² = 0.33, Test R² = 0.14 → Modest overfitting, but acceptable
- **Model B**: Train R² = 0.09, Test R² = **-0.13** → Complete failure, predicts worse than mean

---

### **3.3 RMSE (Root Mean Squared Error)**
**What it is**: Average size of prediction errors, in the same units as the outcome.

**Formula**: `RMSE = √[Σ(Actual - Predicted)² / N]`

**Interpretation**:
- Lower RMSE = better predictions
- RMSE in same units as outcome (number of cases)

**Example from Model A**:
- Train RMSE = 1.5192 → On average, predictions are off by ±1.5 cases
- Test RMSE = 1.7576 → On average, predictions are off by ±1.8 cases

**Why RMSE matters**:
- R² tells you **proportion** of variance explained
- RMSE tells you **absolute size** of errors
- RMSE heavily penalizes large errors (because of squaring)

**Problem with RMSE in our context**:
- With median = 0 cases, being off by 1.8 cases is actually quite large
- USA residual (+819) dominates the RMSE calculation

---

### **3.4 Negative R² Problem**
**What it means**: Model predictions are **worse than just predicting the mean** for all observations.

**How is this possible?**
- R² can be negative on test data
- Means model learned patterns in training data that don't generalize

**Formula**:
- Mean baseline: Predict every country has the mean number of cases
- If model's errors > mean baseline's errors → R² < 0

**Example: Model B**
- Mean ISDS cases ≈ 2.5
- If model predicts wildly wrong values (e.g., predicts 10 for a country with 0), errors are larger than just predicting 2.5 for everyone
- Result: R² = -0.13

**Implications**:
- Model is useless for prediction
- Predictors have no meaningful relationship with outcome
- Should not be used for inference or policy

---

### **3.5 Observations per Predictor Ratio**
**What it is**: Number of data points divided by number of predictors.

**Why it matters**: Models need enough data to reliably estimate coefficients.

**Rules of thumb**:
- **< 5 obs/predictor**: Severe overfitting risk
- **5-10 obs/predictor**: Marginal, use with caution
- **10-20 obs/predictor**: Acceptable
- **> 20 obs/predictor**: Good

**Our analysis**:
- **Minimal version (5 predictors)**:
  - Model A: 50 countries / 5 predictors = **10.0** (acceptable)
  - Model B: 113 countries / 5 predictors = **22.6** (good)

- **Revised version (7 predictors)**:
  - Model A: 23 countries / 7 predictors = **3.3** (severe risk!)
  - Model B: 72 countries / 7 predictors = **10.3** (acceptable)

**Why we chose minimal**: Better statistical properties, less overfitting risk.

---

## 4. Data Challenges & Solutions

### **4.1 Zero-Inflation**
**Definition**: When the outcome variable has an excess of zero values beyond what a standard distribution would predict.

**In our data**:
- **Model A**: 24 out of 50 countries (48%) have **zero** climate cases
- **Model B**: Even worse distribution
- Median = 0 for both outcomes

**Why it's a problem**:
1. Violates normality assumption of OLS
2. Linear regression can predict negative values (impossible for counts)
3. Two distinct processes:
   - **Process 1**: Will a country have ANY cases? (binary: yes/no)
   - **Process 2**: IF they have cases, how many? (count: 1, 2, 3...)

**Standard OLS treats these as one process** → poor fit.

---

### **4.2 Hurdle Models (Solution to Zero-Inflation)**
**What they are**: Two-stage models that separately model:
1. **Stage 1 (Hurdle)**: Probability of having any cases (logistic regression)
2. **Stage 2 (Count)**: Number of cases, given at least one (truncated count model)

**Analogy**:
- Stage 1: Will you go to the store? (yes/no)
- Stage 2: If you go, how many items will you buy? (count)

**How we implemented it**:
- **Stage 1**: Logistic regression predicting `Has_Cases = 1` vs. `Has_Cases = 0`
- **Stage 2**: Linear regression on subset of countries with `cases > 0`

**Results**:
- **Model A, Stage 1**: 85.7% accuracy (excellent!)
  - Predictors of having ANY cases: population, press freedom, number of deals
- **Model A, Stage 2**: Still struggles to predict exact count
  - Once a country has cases, predicting how many remains difficult

**Key insight**: The relationship is **qualitative** (cases vs. no cases) more than **quantitative** (how many cases).

---

### **4.3 Outliers**
**Definition**: Data points that are extremely different from the rest.

**In our data**:
- **USA in Model A**: 969 climate cases (next highest: ~60)
- USA has 16× more cases than the next country
- Creates a **leverage point** that dominates the model

**How outliers affect models**:
1. **Pull the regression line** toward themselves
2. **Inflate R²** on training data (model fits the outlier well)
3. **Reduce generalizability** (model learns "USA pattern" not "general pattern")

**Detection methods**:
- **Visual**: Scatter plots, box plots
- **Statistical**:
  - Standardized residuals > 3
  - Cook's Distance > 1
  - Leverage values > 2(p+1)/n

**What we did**:
- Compared Model A **with USA** vs. **without USA**
- Without USA: Test R² collapsed from +0.14 to **-0.17**
- Conclusion: Model A's "success" is entirely USA-driven

**Outlier handling options**:
1. **Remove**: Justified if data error or truly exceptional case (USA?)
2. **Transform**: Log transformation (but USA still extreme)
3. **Robust regression**: Less sensitive to outliers (not implemented)
4. **Keep but report**: Acknowledge limitation (our approach)

---

### **4.4 Skewed Distributions**
**Definition**: When data is not symmetrically distributed around the mean.

**Types**:
- **Right-skewed (positive)**: Long tail on the right (most values low, few very high)
- **Left-skewed (negative)**: Long tail on the left

**In our data**:
- **Outcome variables**: Heavily right-skewed (median = 0, mean = 2-5, max = 969)
- **Total_Deal_Size**: Right-skewed (few countries with massive deals)
- **Population**: Right-skewed (few countries with billions of people)

**Why it matters**:
- OLS assumes normally distributed residuals
- Skewness violates this assumption
- Results in poor predictions and unreliable significance tests

**Common fixes**:
1. **Log transformation**: `log(x + 1)` to handle zeros
   - We did this for population
   - Could do for outcome, but loses interpretability
2. **Square root transformation**: Less aggressive than log
3. **Use count models**: Poisson or negative binomial regression (not used)

---

### **4.5 Missing Data**
**Definition**: When some observations lack values for certain variables.

**Patterns**:
1. **MCAR (Missing Completely at Random)**: Missingness unrelated to anything
2. **MAR (Missing at Random)**: Missingness related to observed variables
3. **MNAR (Missing Not at Random)**: Missingness related to the missing value itself

**In our data**:
- **Literacy_Rate_Pct**: 74% coverage in Model A → excluded 13 countries
- **Prop_Agriculture**: 52% coverage in Model A → excluded 24 countries
- **Why missing**: Smaller/developing countries have less data collection

**This is likely MAR/MNAR**: Poorer countries (which might have more land deals) have less data.

**Handling strategies**:
1. **Listwise deletion** (our approach): Remove any observation with missing values
   - Pro: Simple, unbiased if MCAR
   - Con: Loses data, reduces sample size, biased if MAR/MNAR
2. **Imputation**: Fill in missing values
   - Mean imputation (simple but biased)
   - Multiple imputation (complex but better)
3. **Exclude variable**: If too much missingness (what we did with Literacy and Prop_Agriculture)

**Trade-off**:
- **Revised model** (7 predictors): More variables, but N=23 (Model A)
- **Minimal model** (5 predictors): Fewer variables, but N=50 (Model A)
- We chose minimal for better statistical power

---

### **4.6 Sample Size Limitations**
**The fundamental challenge**: We have limited countries with complete data.

**Why small samples are problematic**:
1. **Overfitting**: Model memorizes noise instead of learning signal
2. **Unstable coefficients**: Small changes in data → big changes in estimates
3. **Wide confidence intervals**: Less precise estimates
4. **Low statistical power**: Harder to detect true effects

**Our sample sizes**:
- **Model A**: 50 countries total (40 train, 10 test)
- **Model B**: 113 countries total (90 train, 23 test)

**Why we can't just "get more data"**:
- Limited by number of countries in the world (~195)
- Further limited by data availability (land deals, litigation cases, governance indicators)
- This is a **fixed universe** problem, not a sampling problem

**Implications**:
- Must be conservative in model complexity (fewer predictors)
- Results are exploratory, not definitive
- Larger test R² drops are expected and acceptable

---

## 5. Advanced Techniques Used

### **5.1 Standardization (Z-score Normalization)**
**What it is**: Transforming variables to have mean = 0 and standard deviation = 1.

**Formula**: `z = (x - mean) / SD`

**Why we do it**:
1. **Comparability**: Can compare coefficients across variables with different units
   - Example: Compare effect of "deal size in hectares" vs. "corruption score (0-100)"
2. **Numerical stability**: Prevents computational issues with very large/small numbers
3. **Interpretation**: Coefficient = change in outcome for 1 SD increase in predictor

**Example**:
- **Unstandardized**: 1,000,000 more hectares → 0.0001 more cases (tiny coefficient)
- **Standardized**: 1 SD increase in deal size → 0.374 fewer cases (interpretable)

**How to interpret standardized coefficients**:
- Coefficient magnitude = relative importance
- Larger |coefficient| = more important predictor

**In our analysis**: All predictors are standardized before regression.

---

### **5.2 Train-Test Split**
**Purpose**: Evaluate model performance on unseen data.

**How it works**:
1. Randomly split data into training (80%) and test (20%) sets
2. Fit model on training data only
3. Evaluate on test data to see if it generalizes

**Why random split?**:
- Ensures test set is representative
- Avoids bias (e.g., putting all small countries in test set)

**Our implementation**:
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,    # 20% for testing
    random_state=42   # Reproducible results
)
```

**Limitations**:
- With small samples (N=50), test set is only 10 countries
- One unusual country in test set can drastically change Test R²
- More robust: **k-fold cross-validation** (not used)

---

### **5.3 Cross-Validation (Not Used, But Could Be)**
**What it is**: More robust version of train-test split.

**How it works (5-fold example)**:
1. Split data into 5 equal parts (folds)
2. Train on 4 folds, test on 1 fold → Record performance
3. Rotate: Train on different 4 folds, test on remaining fold
4. Repeat 5 times (each fold is test set once)
5. Average performance across all 5 folds

**Advantages over single train-test split**:
- Uses all data for both training and testing
- More stable performance estimates
- Better for small samples

**Why we didn't use it**:
- More complex to implement
- Results harder to report (no single train/test R²)
- Standard in academic social science to use single split

**Recommendation**: Should use 5-fold or 10-fold CV for more robust estimates.

---

### **5.4 Interaction Terms**
**What they are**: Variables that capture how the effect of one predictor **depends on** another predictor.

**Formula**: `interaction = X₁ × X₂`

**Example in our analysis**:
- **Corruption_Score × Num_Deals**: Does the effect of land deals depend on governance quality?
- Hypothesis: In corrupt countries, more deals lead to more litigation (weak rule of law enables both)
- Hypothesis: In clean countries, fewer deals but might still have litigation (strong civil society)

**How to interpret**:
- Main effect: Effect when other variable = 0
- Interaction: How much the effect changes for 1 unit increase in other variable

**Results in our analysis**:
- Interaction terms added **0.8%** to R² (minimal improvement)
- Not worth the added complexity

**When interactions matter**:
- Theory predicts conditional effects
- Scatterplots show different patterns in subgroups
- Our case: Theory suggests interactions, but data doesn't support

---

### **5.5 Logistic Regression (Used in Hurdle Model Stage 1)**
**What it is**: Regression for **binary outcomes** (yes/no, 0/1).

**Difference from linear regression**:
- **Linear**: Predicts continuous values (-∞ to +∞)
- **Logistic**: Predicts probability (0 to 1)

**Formula**: `P(Y=1) = 1 / (1 + e^-(β₀ + β₁X₁ + ...))`

**In our context**:
- Stage 1 of hurdle model: Predict `Has_Cases = 1` vs. `Has_Cases = 0`
- Output: Probability that a country has any cases

**Interpretation of coefficients**:
- **Log-odds scale** (default, hard to interpret)
- **Odds ratios** (exponentiated coefficients, easier)
  - OR > 1: Increases odds of outcome
  - OR < 1: Decreases odds of outcome

**Example from Model A hurdle**:
- log_Population coefficient = +1.63 (log-odds)
- Odds ratio = e^1.63 = 5.1
- Interpretation: Each 1 SD increase in log population **multiplies odds** of having cases by 5.1×

**Performance metric**: Accuracy (% correctly classified)
- Model A Stage 1: 85.7% accuracy (good!)

---

### **5.6 Coefficient Significance Testing**
**What it tests**: Is the coefficient reliably different from zero?

**Null hypothesis**: β = 0 (predictor has no effect)

**Test statistic**: t = (β_estimated - 0) / SE(β)

**P-value interpretation**:
- **p < 0.05**: Reject null, coefficient is "statistically significant"
- **p ≥ 0.05**: Fail to reject null, coefficient not reliably different from zero

**Confidence intervals**:
- 95% CI = β ± 1.96 × SE(β)
- If CI excludes zero → significant

**Important caveats**:
1. **Significance ≠ importance**: A tiny effect can be significant with large N
2. **P-hacking risk**: Testing many models inflates false positive rate
3. **Small samples**: Wide CIs, harder to reach significance (Type II error risk)

**In our reports**: We focus more on **effect sizes** (coefficients) than p-values.

---

## 6. What Could Be Done With This Data

### **6.1 Alternative Outcome Modeling**

#### **A. Binary Classification**
**Instead of predicting count of cases, predict presence/absence**

**Approach**:
- Create binary outcome: `Has_Climate_Cases = 1` if cases > 0, else 0
- Use logistic regression instead of OLS

**Advantages**:
- Addresses zero-inflation directly
- Better suited for our data distribution
- Easier to interpret: "What predicts litigation activity?"

**Models to try**:
1. **Logistic Regression**: Standard approach (we did this in hurdle model)
2. **Decision Trees**: Non-linear relationships, easy to visualize
3. **Random Forest**: Ensemble of trees, handles outliers better
4. **Gradient Boosting** (XGBoost, LightGBM): Often best performance
5. **Support Vector Machines (SVM)**: Good for small samples

**Evaluation metrics**:
- Accuracy: % correct predictions
- Precision: Of predicted positives, how many are correct?
- Recall: Of actual positives, how many did we find?
- AUC-ROC: Overall discrimination ability (0.5 = random, 1.0 = perfect)
- F1-score: Harmonic mean of precision and recall

---

#### **B. Count Models**
**Proper models for count outcomes**

**1. Poisson Regression**
- Assumes: Mean = Variance
- Good for counts with moderate dispersion
- Link function: log(λ) = β₀ + β₁X₁ + ...
- Interpretation: Coefficients are log(rate ratios)

**2. Negative Binomial Regression**
- Allows: Variance > Mean (overdispersion)
- Better for our data (high variance due to USA outlier)
- More flexible than Poisson

**3. Zero-Inflated Poisson (ZIP)**
- Explicitly models excess zeros
- Two parts:
  - Logistic model: Probability of "structural zero" (never can have cases)
  - Poisson model: Count if not structural zero
- Best for our zero-inflation problem

**4. Zero-Inflated Negative Binomial (ZINB)**
- Combines ZIP + overdispersion handling
- Most flexible option
- Likely the best fit for our data

**Implementation**:
```python
from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP

model = ZeroInflatedNegativeBinomialP(
    y,  # outcome
    X,  # predictors
    exog_infl=X  # predictors for zero-inflation
)
results = model.fit()
```

---

#### **C. Ordinal Outcomes**
**Treat case counts as ordered categories**

**Approach**:
- Instead of 0, 1, 2, 3, ..., 969 cases
- Create categories: "None" (0), "Low" (1-5), "Medium" (6-20), "High" (21+)
- Use ordinal logistic regression

**Advantages**:
- Reduces influence of extreme outliers (USA becomes "High" not "969")
- Assumes only ordering matters, not exact distances
- Robust to measurement error in case counts

**Model**: Proportional odds logistic regression
```python
from statsmodels.miscmodels.ordinal_model import OrderedModel

model = OrderedModel(
    y_ordinal,  # categorical outcome
    X,
    distr='logit'
)
```

---

### **6.2 Unsupervised Learning**

#### **A. Clustering Countries**
**Goal**: Find natural groupings of countries based on land deal and governance patterns.

**Methods**:

**1. K-Means Clustering**
- Partitions countries into K groups
- Minimizes within-cluster variance
- Fast, interpretable

**2. Hierarchical Clustering**
- Creates dendrogram (tree) of nested clusters
- Can visualize relationships between country groups
- Doesn't require pre-specifying K

**3. DBSCAN**
- Density-based clustering
- Automatically identifies outliers (USA!)
- Doesn't assume spherical clusters

**Example research questions**:
- Do countries cluster into distinct "profiles" (e.g., "large deals + corrupt", "small deals + free press")?
- Are litigation patterns different across clusters?
- Can we identify "at-risk" profiles for future litigation?

**Implementation**:
```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Analyze: Which cluster has most litigation?
```

**Value**: Could reveal hidden country typologies not captured by regression.

---

#### **B. Principal Component Analysis (PCA)**
**Goal**: Reduce many correlated predictors into fewer uncorrelated components.

**How it works**:
- Find linear combinations of predictors that capture maximum variance
- PC1 (first component) explains most variance
- PC2 explains second most, orthogonal to PC1
- Etc.

**Example for our data**:
- We have governance variables: Corruption, Press Freedom, Literacy
- These are correlated (corrupt countries tend to have less press freedom)
- PCA could create a single "Governance Quality" component

**Advantages**:
- Reduces multicollinearity
- Improves obs/predictor ratio
- May reveal latent constructs

**Disadvantages**:
- Loses interpretability (PC1 = 0.6×Corruption + 0.4×Press + 0.3×Literacy - what is it?)
- May not align with theoretical concepts

**When useful**: Exploratory analysis, high-dimensional data, strict multicollinearity.

---

#### **C. Dimensionality Reduction for Visualization**
**Goal**: Visualize high-dimensional country profiles in 2D/3D.

**t-SNE (t-distributed Stochastic Neighbor Embedding)**
- Projects high-dimensional data to 2D while preserving local structure
- Good for visualizing clusters
- Non-linear, preserves neighborhoods

**UMAP (Uniform Manifold Approximation and Projection)**
- Similar to t-SNE but faster and more scalable
- Better preserves global structure
- State-of-the-art for visualization

**Application**:
```python
from umap import UMAP

reducer = UMAP(n_components=2, random_state=42)
embedding = reducer.fit_transform(X_scaled)

# Scatter plot colored by litigation cases
plt.scatter(embedding[:, 0], embedding[:, 1],
            c=y, cmap='viridis', s=50)
plt.colorbar(label='Litigation Cases')
```

**Insight**: See if countries with similar land deal/governance profiles have similar litigation patterns.

---

### **6.3 Feature Engineering**

#### **A. Create New Predictors**

**1. Ratio Variables**
- **Deal_Size_per_Deal** = Total_Deal_Size / Num_Deals
  - Captures average deal size
  - May predict differently than total size

- **Cases_per_Million_Pop** = Cases / (Population / 1,000,000)
  - Controls for population in the outcome instead of predictor
  - Litigation rate per capita

- **Land_per_Capita** = Total_Deal_Size / Population
  - How much land acquired per person
  - May capture "intensity" of land pressure

**2. Interaction Features**
- **Weak_Governance × Large_Deals**: Are large deals especially problematic in corrupt contexts?
- **Press_Freedom × Population**: Do large countries with free press have disproportionately more cases?

**3. Temporal Features** (if time-series data available)
- **Years_Since_First_Deal**: How long has land acquisition been happening?
- **Deal_Growth_Rate**: Are deals increasing or decreasing?
- **Litigation_Lag**: Time between deals and cases emerging

**4. Geographic Features**
- **Region** (Africa, Asia, Latin America, etc.): Dummy variables or clustering
- **Neighboring_Countries_Cases**: Spatial spillover effects
- **Landlocked**: Different land dynamics?

**5. External Indices**
- **Democracy Index**: From Economist Intelligence Unit
- **Rule of Law Index**: From World Justice Project
- **Human Development Index**: Composite of health, education, income
- **Conflict/Fragility Indices**: Post-conflict countries may have more land disputes

---

#### **B. Polynomial Features**
**Capturing non-linear relationships**

**Example**:
- Instead of just `Total_Deal_Size`
- Include `Total_Deal_Size²`
- Allows U-shaped or inverted-U relationships

**Hypothesis for land deals**:
- Small deals → Few cases (not enough to mobilize)
- Medium deals → More cases (visible harm, mobilization possible)
- Very large deals → Fewer cases again (government repression, stakeholder capture)
- This is a **non-monotonic relationship** that linear regression can't capture

**Implementation**:
```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
```

---

#### **C. Categorical Encoding**
**If categorical variables are available**

**Examples**:
- **Investor_Origin**: Is the investor domestic or foreign? Which country?
- **Sector**: Agriculture, mining, forestry, biofuels
- **Deal_Status**: Concluded, ongoing, failed
- **Legal_System**: Common law, civil law, mixed

**Encoding methods**:
1. **One-Hot Encoding**: Create binary dummy variables for each category
   - Sector_Agriculture = 1 or 0
   - Sector_Mining = 1 or 0
   - Etc.

2. **Target Encoding**: Replace category with mean outcome for that category
   - E.g., "Agriculture" → 3.5 (average cases for ag deals)
   - Risk of overfitting, need cross-validation

3. **Ordinal Encoding**: If categories have natural order
   - E.g., Deal_Status: Failed=0, Ongoing=1, Concluded=2

---

### **6.4 Advanced Machine Learning Models**

#### **A. Ensemble Methods**

**1. Random Forest**
**What it is**: Collection of decision trees, each trained on random subset of data and features.

**Advantages**:
- Handles non-linear relationships
- Robust to outliers (USA won't dominate)
- No multicollinearity issues
- Provides feature importance rankings
- Little hyperparameter tuning needed

**Disadvantages**:
- Less interpretable (black box)
- Can't extrapolate beyond training data range
- May overfit with small samples (our case)

**Implementation**:
```python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=100,      # Number of trees
    max_depth=5,           # Prevent overfitting
    min_samples_split=5,   # Prevent overfitting
    random_state=42
)
rf.fit(X_train, y_train)

# Feature importance
importance = pd.DataFrame({
    'feature': feature_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
```

**Best for**:
- Exploratory analysis
- Feature selection (drop unimportant features)
- Benchmark for linear model performance

---

**2. Gradient Boosting (XGBoost, LightGBM, CatBoost)**
**What it is**: Sequentially builds trees, each correcting errors of previous trees.

**Advantages**:
- Often best predictive performance
- Handles mixed data types
- Built-in regularization
- Less overfitting than random forest (with proper tuning)

**Disadvantages**:
- Requires careful hyperparameter tuning
- Sensitive to outliers (can overfit to USA)
- Computationally intensive
- Black box (hard to interpret)

**When to use**: Prediction competitions, when accuracy is paramount over interpretation.

**Example**:
```python
import xgboost as xgb

model = xgb.XGBRegressor(
    n_estimators=50,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)
```

---

**3. Stacking**
**What it is**: Combine predictions from multiple models using a meta-model.

**How it works**:
1. Train several base models (e.g., linear regression, random forest, XGBoost)
2. Use their predictions as features for a meta-model (e.g., ridge regression)
3. Meta-model learns optimal weighting of base models

**Advantages**:
- Leverages strengths of different model types
- Often improves accuracy
- Can combine interpretable + black-box models

**Example**:
```python
from sklearn.ensemble import StackingRegressor

estimators = [
    ('ols', LinearRegression()),
    ('rf', RandomForestRegressor(n_estimators=50, random_state=42)),
    ('xgb', xgb.XGBRegressor(n_estimators=50, random_state=42))
]

stacking = StackingRegressor(
    estimators=estimators,
    final_estimator=Ridge(alpha=1.0)
)
stacking.fit(X_train, y_train)
```

---

#### **B. Regularization (Preventing Overfitting)**

**Problem**: With small samples, models easily overfit.

**Solution**: Penalize complex models (large coefficients).

**1. Ridge Regression (L2 Regularization)**
- Adds penalty: `λ × Σ(β²)`
- Shrinks all coefficients toward zero
- Doesn't eliminate predictors (all stay in model)

**2. Lasso Regression (L1 Regularization)**
- Adds penalty: `λ × Σ|β|`
- Can shrink coefficients exactly to zero
- Performs automatic feature selection

**3. Elastic Net**
- Combines L1 + L2: `λ₁ × Σ|β| + λ₂ × Σ(β²)`
- Best of both worlds

**In our context**:
- With 5-10 predictors, regularization less critical
- Could help with Model A (small sample, N=50)
- Lasso could identify which predictors truly matter

**Implementation**:
```python
from sklearn.linear_model import LassoCV

# Cross-validated to find optimal penalty
lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X_train, y_train)

# Which predictors survived?
coef_df = pd.DataFrame({
    'feature': feature_names,
    'coefficient': lasso.coef_
})
print(coef_df[coef_df.coefficient != 0])
```

---

### **6.5 Time-Series Analysis** (If Temporal Data Available)

#### **A. Panel Data Models**
**If we have cases/deals over multiple years per country**

**Fixed Effects Model**:
- Controls for time-invariant country characteristics
- Each country is its own control
- Question: Does an increase in land deals **within a country over time** predict more cases?

**Random Effects Model**:
- Assumes country effects are random draws from a distribution
- More efficient if assumption holds

**Difference-in-Differences**:
- Compare countries that experienced large deal increase vs. those that didn't
- Before-after comparison
- Causal interpretation possible

**Example**:
```python
from linearmodels.panel import PanelOLS

# Data must be multi-indexed by country and year
model = PanelOLS(
    y,  # Cases
    X,  # Predictors
    entity_effects=True  # Country fixed effects
)
results = model.fit()
```

---

#### **B. Lagged Predictors**
**Land deals today → Litigation in 5 years**

**Approach**:
- Create lagged variables: `Num_Deals_t-1`, `Num_Deals_t-2`, etc.
- Predict current litigation from past land deals
- Tests causal sequencing (deals must precede cases)

**Distributed Lag Models**:
- Include multiple lags: Effect of deals 1 year ago, 2 years ago, etc.
- Cumulative effect over time

**Granger Causality**:
- Tests if past values of X help predict future Y
- Not true causality, but suggests temporal precedence

---

#### **C. Survival Analysis**
**Time until first litigation case**

**Question**: How long after land deals begin does litigation emerge?

**Kaplan-Meier Curves**:
- Plot probability of remaining "litigation-free" over time
- Compare across country groups (high vs. low corruption)

**Cox Proportional Hazards**:
- Regression model for time-to-event data
- Predictors: Land deal size, governance quality, etc.
- Outcome: Hazard rate (risk of litigation at time t)

**Example**:
```python
from lifelines import CoxPHFitter

cph = CoxPHFitter()
cph.fit(
    df,
    duration_col='years_to_first_case',
    event_col='case_occurred',  # 1 if case happened, 0 if censored
    formula='Total_Deal_Size + Corruption_Score + ...'
)
cph.print_summary()
```

---

### **6.6 Causal Inference Methods**

#### **A. Instrumental Variables (IV)**
**Problem**: Land deals and litigation may both be caused by a third factor (e.g., resource curse, weak institutions).

**Solution**: Find an **instrument** - a variable that:
1. Affects land deals (relevance)
2. Only affects litigation through land deals (exclusion restriction)
3. Is uncorrelated with the error term (exogeneity)

**Possible instruments**:
- **Global commodity prices**: High prices → more land deals, but doesn't directly cause litigation
- **Foreign aid flows**: Aid → infrastructure → land accessibility → more deals
- **Investor country GDP**: Rich investor countries → more outward FDI → more deals

**Method**: Two-Stage Least Squares (2SLS)
```python
from linearmodels.iv import IV2SLS

model = IV2SLS(
    dependent=y,  # Litigation cases
    exog=X_controls,  # Corruption, press freedom, population
    endog=X_endogenous,  # Land deals (endogenous predictor)
    instruments=Z  # Instrument (commodity prices)
)
results = model.fit()
```

**Interpretation**: Estimates **causal effect** of land deals on litigation (under IV assumptions).

---

#### **B. Propensity Score Matching**
**Question**: Compare countries with many land deals to similar countries with few deals.

**How it works**:
1. Estimate probability of having many land deals based on observable characteristics (propensity score)
2. Match treated (many deals) to control (few deals) countries with similar propensity scores
3. Compare litigation rates between matched pairs

**Advantages**:
- Reduces confounding from observed variables
- Transparent matching process
- Robust to model misspecification

**Disadvantages**:
- Can't control for unobserved confounders
- Requires large sample for good matches (we're borderline)

**Example**:
```python
from sklearn.neighbors import NearestNeighbors

# 1. Estimate propensity scores
from sklearn.linear_model import LogisticRegression
ps_model = LogisticRegression()
ps_model.fit(X_controls, treatment)  # treatment = high vs. low deals
propensity_scores = ps_model.predict_proba(X_controls)[:, 1]

# 2. Match on propensity scores
matcher = NearestNeighbors(n_neighbors=1)
matcher.fit(propensity_scores[treatment==0].reshape(-1,1))
matches = matcher.kneighbors(propensity_scores[treatment==1].reshape(-1,1))

# 3. Compare outcomes in matched sample
```

---

#### **C. Regression Discontinuity**
**If there's a threshold for land deals**

**Example scenario**: Countries must approve land deals >50,000 hectares through parliament.

**Design**:
- Compare countries just below threshold (49,999 ha) to just above (50,001 ha)
- Assume they're similar except for treatment (parliamentary approval)
- Difference in litigation = causal effect of approval process

**Strong causal inference**: Only works if threshold creates quasi-random assignment.

**Our data**: Likely no such thresholds, but worth exploring policy discontinuities.

---

### **6.7 Network Analysis** (If Relational Data Available)

#### **A. Investor Networks**
**If we know which investors are involved in which deals**

**Questions**:
- Do investors cluster into networks?
- Are certain investors associated with more litigation?
- Do countries targeted by the same investors have similar litigation patterns?

**Methods**:
- **Bipartite network**: Countries ↔ Investors
- **Community detection**: Groups of countries with shared investors
- **Centrality measures**: Which investors are most "central" in land deal networks?

**Example**:
```python
import networkx as nx

# Create bipartite network
G = nx.Graph()
G.add_nodes_from(countries, bipartite=0)
G.add_nodes_from(investors, bipartite=1)
G.add_edges_from(deals)  # (country, investor) pairs

# Find communities
from networkx.algorithms import community
communities = community.louvain_communities(G)
```

---

#### **B. Spatial Networks**
**Neighboring countries**

**Question**: Does litigation in neighboring countries predict litigation elsewhere? (Spillover effects)

**Spatial lag model**:
- Y_i = β₀ + ρ × W × Y + β₁X₁ + ... + ε
- W = spatial weight matrix (neighbors)
- ρ = spatial correlation parameter

**Applications**:
- Legal precedents diffuse across borders
- Regional civil society coordination
- Shared investor concerns

**Software**: `PySAL`, `GeoPandas`

---

### **6.8 Text Analysis** (If Case Text Available)

#### **A. Topic Modeling**
**If we have text of litigation cases**

**Goal**: Discover common themes/topics in cases.

**Method**: Latent Dirichlet Allocation (LDA)
- Unsupervised: Finds K topics from word patterns
- Each case is a mixture of topics
- Each topic is a mixture of words

**Example topics**:
- Topic 1: "water rights, irrigation, downstream, pollution" → Water disputes
- Topic 2: "indigenous, customary, tenure, displacement" → Land rights
- Topic 3: "investment, expropriation, compensation, treaty" → ISDS

**Application**:
- Do land deal characteristics predict which topics appear in cases?
- Are certain topics more common in countries with corrupt governance?

**Implementation**:
```python
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(max_features=1000, stop_words='english')
X_text = vectorizer.fit_transform(case_texts)

lda = LatentDirichletAllocation(n_components=5, random_state=42)
topics = lda.fit_transform(X_text)
```

---

#### **B. Sentiment Analysis**
**If we have news coverage or case documents**

**Goal**: Measure public/media sentiment about land deals.

**Methods**:
- **Lexicon-based**: Count positive/negative words (VADER, TextBlob)
- **ML-based**: Train classifier on labeled sentiment data (BERT, GPT)

**Questions**:
- Does negative media coverage predict litigation?
- Are deals in countries with free press more likely to have negative coverage?

---

#### **C. Named Entity Recognition (NER)**
**Extract structured info from unstructured text**

**Entities to extract**:
- Companies involved
- Government agencies
- Affected communities
- Legal frameworks cited

**Application**: Build structured dataset from case law for more detailed analysis.

---

### **6.9 Explanatory Model Interpretation**

#### **A. SHAP (SHapley Additive exPlanations)**
**For complex models (random forest, XGBoost)**

**What it does**: Explains each prediction by attributing contribution of each feature.

**Example**:
- USA has 969 cases (predicted: 150)
- SHAP shows: Population contributes +200, Num_Deals contributes +150, etc.

**Benefits**:
- Model-agnostic (works with any model)
- Theoretically grounded (Shapley values from game theory)
- Visualizations: Force plots, summary plots, dependence plots

**Implementation**:
```python
import shap

explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# Visualize
shap.plots.waterfall(shap_values[0])  # Individual prediction
shap.plots.beeswarm(shap_values)      # Overall feature importance
```

---

#### **B. LIME (Local Interpretable Model-agnostic Explanations)**
**Explain individual predictions**

**How it works**:
1. Perturb input features around a single observation
2. Fit simple linear model to perturbed samples
3. Use linear model to explain that one prediction

**Use case**: "Why did the model predict USA would have 150 cases?"

**Implementation**:
```python
from lime.lime_tabular import LimeTabularExplainer

explainer = LimeTabularExplainer(
    X_train,
    feature_names=feature_names,
    mode='regression'
)

explanation = explainer.explain_instance(
    X_test[0],  # USA
    model.predict
)
explanation.show_in_notebook()
```

---

#### **C. Partial Dependence Plots (PDP)**
**Show effect of one predictor across its range**

**What it shows**:
- X-axis: Values of predictor (e.g., Corruption Score 0-100)
- Y-axis: Average predicted litigation cases at that value

**Benefits**:
- Visualizes marginal effect of predictor
- Shows non-linearities
- Works with any model

**Example**:
```python
from sklearn.inspection import PartialDependenceDisplay

PartialDependenceDisplay.from_estimator(
    model,
    X_train,
    features=[0, 1, 2],  # First 3 predictors
    feature_names=feature_names
)
```

---

### **6.10 Robustness Checks & Sensitivity Analysis**

#### **A. Leave-One-Out Country Analysis**
**Test sensitivity to individual countries**

**Method**:
1. Fit model on all data
2. Re-fit excluding Country 1, check if coefficients change
3. Repeat for Country 2, ..., Country N
4. Identify influential countries

**We did this for USA**: Removing USA collapsed Model A's Test R².

**Should also do for**:
- Brazil (large number of deals)
- India (large population)
- Other potential outliers

---

#### **B. Bootstrap Confidence Intervals**
**Assess uncertainty in estimates**

**Method**:
1. Resample data with replacement (create 1000 bootstrap samples)
2. Fit model on each sample
3. Record coefficients
4. 95% CI = 2.5th to 97.5th percentile of bootstrap coefficients

**Benefits**:
- No assumptions about distribution
- Works with small samples
- Captures full uncertainty

**Implementation**:
```python
from sklearn.utils import resample

bootstrap_coefs = []
for i in range(1000):
    X_boot, y_boot = resample(X_train, y_train, random_state=i)
    model.fit(X_boot, y_boot)
    bootstrap_coefs.append(model.coef_)

# Confidence intervals
ci_lower = np.percentile(bootstrap_coefs, 2.5, axis=0)
ci_upper = np.percentile(bootstrap_coefs, 97.5, axis=0)
```

---

#### **C. Alternative Specifications**
**Test robustness to modeling choices**

**Variations to try**:
1. **Different predictor sets**:
   - Add Literacy and Prop_Agriculture back in (revised model)
   - Remove governance variables (only land deal predictors)
   - Only governance variables (no land deals)

2. **Different transformations**:
   - Log(Cases + 1) instead of raw cases
   - Sqrt(Cases) transformation
   - Winsorize outcome (cap extreme values)

3. **Different samples**:
   - Exclude all outliers (USA, Brazil, India)
   - Only countries with at least 1 case
   - Only countries above median deal size

4. **Different model types**:
   - Negative binomial instead of OLS
   - Hurdle model instead of standard regression
   - Robust regression instead of OLS

**Report**:
- Are core findings consistent across specifications?
- If yes → robust
- If no → findings are fragile

---

## 7. Recommended Next Steps

### **7.1 Immediate Priorities** (Low-Hanging Fruit)

#### **1. Implement Zero-Inflated Negative Binomial**
**Why**: Most appropriate model for count data with excess zeros and overdispersion.

**Expected improvement**:
- Better fit than OLS
- Proper treatment of zeros
- More interpretable coefficients (rate ratios)

**Effort**: Medium (requires `statsmodels` or `R`)

---

#### **2. Remove USA and Report Two Models**
**Why**: USA is an extreme outlier driving Model A results.

**Report**:
- Model A1 (All 50 countries): Show current results with caveat
- Model A2 (Excluding USA, N=49): Show robustness

**Expected result**: Model A2 will perform worse, revealing USA-dependence.

**Effort**: Low (already done in enhanced analysis)

---

#### **3. Focus on Binary Outcome**
**Why**: Predicting presence/absence works better (85.7% accuracy) than predicting counts.

**Approach**: Use logistic regression as primary model for Model A.

**Interpretation shift**:
- From: "Land deals predict how many cases"
- To: "Land deals predict whether a country has any litigation activity"

**Effort**: Low (already implemented in hurdle model)

---

#### **4. Feature Importance from Random Forest**
**Why**: Validate which predictors matter most without linear assumptions.

**Benefits**:
- Captures non-linear effects
- Robust to outliers
- Provides clear ranking

**Effort**: Low (quick to implement)

---

### **7.2 Medium-Term Enhancements**

#### **1. Collect Temporal Data**
**What to get**:
- Year of each land deal
- Year of each litigation case
- Annual governance indicators

**Why**: Enable:
- Panel data models (fixed effects)
- Lagged predictors (deals → cases with time lag)
- Causal inference (temporal precedence)

**Effort**: High (requires data mining)
**Payoff**: High (much stronger causal claims possible)

---

#### **2. Sector-Specific Analysis**
**What**: Separate models for agriculture, mining, forestry deals.

**Why**: Different sectors may have different litigation patterns.

**Current finding**: Ag/mining ISDS (Model B) didn't improve performance, but worth exploring for climate cases.

**Effort**: Medium (requires sector coding of deals)

---

#### **3. Regional Sub-Models**
**What**: Separate models for Africa, Asia, Latin America.

**Why**:
- Regional heterogeneity in legal systems
- Different land tenure traditions
- Varying civil society strength

**Trade-off**: Smaller samples per region, but more homogeneous.

**Effort**: Medium

---

#### **4. Investor Characteristics**
**What to collect**:
- Investor origin (domestic vs. foreign)
- Investor type (private, state-owned, sovereign wealth fund)
- Investor country governance

**Hypothesis**:
- Foreign investors → more ISDS cases (treaty protection)
- State-backed investors → less litigation (political protection)

**Effort**: High (requires deal-level data)
**Payoff**: High (new theoretical insights)

---

### **7.3 Advanced Research Directions**

#### **1. Qualitative Comparative Analysis (QCA)**
**What it is**: Set-theoretic method for identifying necessary/sufficient conditions.

**Question**: What combinations of conditions lead to litigation?
- Example: High deals + Low corruption → Cases?
- Example: Low deals + High press freedom → Cases?

**Benefits**:
- Handles complex causality (multiple pathways)
- Small-N appropriate
- Identifies configurational patterns

**Effort**: High (requires `fsQCA` software, theoretical recoding)

---

#### **2. Bayesian Regression**
**What**: Incorporate prior knowledge into model estimation.

**Benefits**:
- Better for small samples (regularizes estimates)
- Provides full uncertainty distribution
- Can incorporate expert knowledge

**Example prior**: "We expect corruption to increase litigation, with coefficient between 0 and 1"

**Software**: `PyMC3`, `Stan`

**Effort**: High (steep learning curve)

---

#### **3. Agent-Based Modeling**
**What**: Simulate micro-level processes to understand macro patterns.

**Model components**:
- Agents: Investors, governments, civil society
- Rules: Investors seek land, civil society responds if mobilization capacity exists
- Emergent outcome: Litigation cases

**Benefits**:
- Test theoretical mechanisms
- Explore counterfactuals
- Generate predictions for out-of-sample scenarios

**Effort**: Very high (requires custom programming)

---

#### **4. Meta-Analysis**
**What**: Combine results across multiple studies/datasets.

**Approach**:
- Find other studies on land deals → litigation
- Standardize effect sizes
- Pool estimates using meta-analytic techniques

**Benefits**:
- Larger effective sample size
- Test generalizability across contexts
- Identify moderators (where effects are stronger/weaker)

**Effort**: High (requires literature review, effect size extraction)

---

### **7.4 Data Collection Priorities**

To substantially improve this analysis, prioritize collecting:

1. **Time-series data** (HIGH PRIORITY)
   - When did each deal occur?
   - When was each case filed?
   - Enables causal inference

2. **Deal-level characteristics** (HIGH PRIORITY)
   - Investor details
   - Sector codes
   - Deal status (concluded, abandoned, renegotiated)
   - Enables nuanced mechanisms testing

3. **Case-level characteristics** (MEDIUM PRIORITY)
   - Plaintiff type (community, NGO, government)
   - Case outcome (win, loss, settled)
   - Issues raised (land rights, environment, labor)
   - Enables outcome prediction

4. **Additional country covariates** (MEDIUM PRIORITY)
   - Democracy indices
   - Judicial independence scores
   - NGO density/capacity
   - Historical conflict data
   - Enables better controls

5. **Subnational data** (LOW PRIORITY, HIGH PAYOFF)
   - Region/province within country
   - Local governance quality
   - Enables within-country analysis (much stronger causal inference)

---

## Summary: Best Path Forward

### **Quick Wins** (Do First)
1. ✓ Binary outcome logistic regression (already done, report as primary)
2. ✓ Exclude USA and show fragility (already done, report transparently)
3. Implement ZINB model (proper count model)
4. Random forest feature importance (validate linear model)

### **Strategic Priorities**
1. Collect temporal data (biggest unlock for causal inference)
2. Focus narrative on qualitative relationship (cases vs. no cases)
3. Acknowledge Model B failure, focus on Model A insights

### **Research Impact**
Current contribution:
- Exploratory evidence of land deals → climate litigation link
- Demonstrates ISDS cases follow different pathways

With recommended enhancements:
- Causal claims about land deals → litigation
- Mechanistic understanding (which types of deals, in which contexts)
- Predictive tool for identifying at-risk countries

---

## Conclusion

This analysis has made excellent progress in exploratory data analysis with limited data. The key findings are:

1. **Relationship exists but is fragile**: Land deals predict climate litigation presence, but not robustly across all contexts
2. **ISDS is fundamentally different**: No evidence land deals predict ISDS cases
3. **Methodological challenges**: Zero-inflation, outliers, and small samples limit inference

**Moving forward**, the analysis would benefit most from:
- Proper count models (ZINB)
- Temporal data for causal claims
- Focus on binary outcomes where evidence is strongest
- Transparent reporting of limitations

The data contains valuable signal, but requires careful statistical handling and realistic expectations about what can be learned from N=50 countries.

---

## Glossary - Quick Reference

| Term | Definition | Example |
|------|------------|---------|
| **R²** | Proportion of variance explained | 0.30 = 30% of variance explained |
| **RMSE** | Average prediction error | 1.5 = off by ±1.5 cases on average |
| **Coefficient (β)** | Effect of 1-unit change in predictor | β=0.5 → +1 deal → +0.5 cases |
| **Residual** | Prediction error | Actual - Predicted |
| **Overfitting** | Model memorizes training data | Train R² >> Test R² |
| **Zero-inflation** | Excess zeros in outcome | 48% of countries have 0 cases |
| **Hurdle model** | Two-stage: any cases? How many? | Stage 1: logistic; Stage 2: count |
| **Standardization** | Scale to mean=0, SD=1 | Allows comparing coefficients |
| **Train-test split** | Evaluate on unseen data | 80% train, 20% test |
| **P-value** | Probability of observing if no effect | p<0.05 = "significant" |
| **CI (Confidence Interval)** | Range of plausible values | 95% CI = [0.2, 0.8] |
| **Multicollinearity** | Predictors highly correlated | VIF > 10 problematic |
| **Outlier** | Extreme value | USA: 969 cases vs. median 0 |
| **Logistic regression** | Binary outcome model | Predict yes/no |
| **Count model** | Model for 0, 1, 2, 3... outcomes | Poisson, negative binomial |

---

**Document maintained by**: Research team
**For questions**: See analysis notebooks or contact project lead
**Last updated**: 2026-01-28
