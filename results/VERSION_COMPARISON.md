   # RQ2 Analysis - Version Comparison

   ## Date: 2026-01-26

   ---

   ## TWO VERSIONS AVAILABLE

   You now have **two complete analysis notebooks** to choose from, each with different trade-offs:

   ### **Version 1: REVISED (7 predictors)**
   **File**: `RQ2_Land_Deals_to_Litigation_Analysis_REVISED.ipynb`

   **Predictor Set (7 variables)**:
   1. Total_Deal_Size
   2. Num_Deals
   3. Prop_Agriculture ← **Includes sector composition**
   4. Corruption_Score
   5. Press_Freedom_Score
   6. Literacy_Rate_Pct ← **Includes education control**
   7. log_Population

   **Sample Sizes**:
   - Model A (Climate): **23 countries**
   - Model B (ISDS): **72 countries**

   **Observations per Predictor**:
   - Model A: **3.3** (marginal)
   - Model B: **10.3** (good)

   **Outputs**: `results/revised/`

   ---

   ### **Version 2: MINIMAL (5 predictors)** ⭐ RECOMMENDED
   **File**: `RQ2_Land_Deals_to_Litigation_Analysis_MINIMAL.ipynb`

   **Predictor Set (5 variables)**:
   1. Total_Deal_Size
   2. Num_Deals
   3. Corruption_Score
   4. Press_Freedom_Score
   5. log_Population

   **Sample Sizes**:
   - Model A (Climate): **50 countries** (+117% vs REVISED)
   - Model B (ISDS): **113 countries** (+57% vs REVISED)

   **Observations per Predictor**:
   - Model A: **10.0** (excellent)
   - Model B: **22.6** (excellent)

   **Outputs**: `results/minimal/`

   ---

   ## COMPARISON TABLE

   | Metric | REVISED (7 pred) | MINIMAL (5 pred) | Winner |
   |--------|------------------|------------------|--------|
   | **Model A Sample** | 23 countries | 50 countries | ✓ MINIMAL |
   | **Model A Obs/Pred** | 3.3 | 10.0 | ✓ MINIMAL |
   | **Model B Sample** | 72 countries | 113 countries | ✓ MINIMAL |
   | **Model B Obs/Pred** | 10.3 | 22.6 | ✓ MINIMAL |
   | **Theoretical Depth** | More controls | Core variables | REVISED |
   | **Statistical Power** | Marginal (A) | Good (both) | ✓ MINIMAL |
   | **Generalizability** | Lower | Higher | ✓ MINIMAL |
   | **Interpretability** | Complex | Cleaner | ✓ MINIMAL |

   ---

   ## WHICH VERSION TO USE?

   ### **Use MINIMAL Version If:**
   - ✓ You want **maximum statistical power**
   - ✓ You want models that **generalize well**
   - ✓ You prioritize **sample size** over additional controls
   - ✓ You want to include **major economies** (USA, CAN, AUS, DEU, FRA, GBR in Model A)
   - ✓ You want **cleaner, more interpretable** results
   - ✓ **RECOMMENDED for publication**

   ### **Use REVISED Version If:**
   - You want to control for **education (Literacy)**
   - You want to include **sector composition (Prop_Agriculture)**
   - You want **more nuanced** predictor set
   - You're okay with **smaller sample** in Model A
   - You're doing **exploratory analysis** before finalizing

   ---

   ## VARIABLES DROPPED IN MINIMAL

   ### Prop_Agriculture
   - **Coverage**: Only 52% in Model A, 34% in Model B
   - **Why dropped**: Structurally missing for countries without land deals
   - **Impact**: Losing this doesn't hurt theory - land deal SIZE and COUNT still capture core mechanism

   ### Literacy_Rate_Pct
   - **Coverage**: 74% in Model A, 90% in Model B
   - **Why dropped**: Missing for 13 countries with climate cases
   - **Impact**: Education control is secondary to governance variables (Corruption, Press Freedom)

   ---

   ## VARIABLES KEPT IN BOTH

   ### Land Deal Variables (2)
   - **Total_Deal_Size**: Total hectares acquired
   - **Num_Deals**: Number of separate deals
   - **Rationale**: Core independent variables testing land grabbing hypothesis

   ### Governance Variables (2)
   - **Corruption_Score**: Governance quality (CPI)
   - **Press_Freedom_Score**: Civil society strength
   - **Rationale**: Theoretical mechanisms - weak governance enables both land deals AND litigation

   ### Control Variable (1)
   - **log_Population**: Country size
   - **Rationale**: Larger countries have more potential for cases

   ---

   ## DIAGNOSTIC COMPARISON

   ### Original Problem (Before Revisions)
   - **Predictors**: 8 (included Rule_of_Law, Avg_Deal_Size)
   - **Model A**: 19 countries, Test R² = **-6.36** ❌ (catastrophic overfitting)
   - **Model B**: 42 countries, Test R² = **-0.07** ❌ (overfitting)

   ### After REVISED Version
   - **Predictors**: 7
   - **Model A**: 23 countries, Test R² = **?** (to be run)
   - **Model B**: 72 countries, Test R² = **?** (to be run)
   - **Expected**: Positive Test R² (0.1-0.4 range)

   ### After MINIMAL Version
   - **Predictors**: 5
   - **Model A**: 50 countries, Test R² = **?** (to be run)
   - **Model B**: 113 countries, Test R² = **?** (to be run)
   - **Expected**: Positive Test R² (0.2-0.5 range), better generalization

   ---

   ## HOW TO RUN BOTH VERSIONS

   ### Run REVISED Version
   ```bash
   cd "/home/caspar/Coding workspaces/Spacial DATASCIENCE/Group-project-Data-Science-in-Space"
   jupyter notebook RQ2_Land_Deals_to_Litigation_Analysis_REVISED.ipynb
   ```

   ### Run MINIMAL Version
   ```bash
   cd "/home/caspar/Coding workspaces/Spacial DATASCIENCE/Group-project-Data-Science-in-Space"
   jupyter notebook RQ2_Land_Deals_to_Litigation_Analysis_MINIMAL.ipynb
   ```

   ---

   ## OUTPUT STRUCTURE

   ```
   results/
   ├── revised/                          # 7-predictor version outputs
   │   ├── merged_country_level_data_revised.csv
   │   ├── modelA_summary_revised.txt
   │   ├── modelB_summary_revised.txt
   │   ├── model_comparison_revised.csv
   │   └── [all figures]_revised.png
   │
   └── minimal/                          # 5-predictor version outputs
      ├── merged_country_level_data_minimal.csv
      ├── modelA_summary_minimal.txt
      ├── modelB_summary_minimal.txt
      ├── model_comparison_minimal.csv
      └── [all figures]_minimal.png
   ```

   ---

   ## RECOMMENDATION

   **Start with MINIMAL version** for the following reasons:

   1. **Statistical validity**:
      - Model A: 10.0 obs/predictor (meets 10+ threshold)
      - Model B: 22.6 obs/predictor (excellent)

   2. **Generalizability**:
      - Larger test sets (10 and 23 countries)
      - More robust Test R² estimates

   3. **Interpretability**:
      - Cleaner predictor set
      - Core theoretical variables only
      - Easier to explain in publications

   4. **Coverage**:
      - Includes ALL 50 countries with climate cases
      - Includes major economies in Model A

   5. **Statistical power**:
      - Better ability to detect real effects
      - Lower risk of Type II errors

   ---

   ## NEXT STEPS

   1. **Run MINIMAL version first** to see actual Test R² performance
   2. **If Model A Test R² > 0.2**: Use MINIMAL version for publication
   3. **If Model A Test R² < 0.2**: Consider:
      - Robustness check with REVISED version
      - Alternative modeling (logistic regression with binary DV)
      - Focus on Model B (larger sample, better performance)
   4. **Compare coefficients** between versions to check stability
   5. **Update INTERPRETATION_AND_ISSUES.md** with new findings

   ---

   ## SUMMARY

   | Version | Best For | Trade-off |
   |---------|----------|-----------|
   | **REVISED** | Exploratory, nuanced controls | Smaller sample, marginal power |
   | **MINIMAL** ⭐ | Publication, generalizability | Fewer controls |

   **Bottom line**: MINIMAL version offers **better statistical properties** while keeping **core theoretical variables**. Use it unless you have strong theoretical reasons to include Literacy and Prop_Agriculture.
