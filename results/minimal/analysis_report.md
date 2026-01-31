# Minimal Model Results: Analysis Report

## Overview

This analysis examines two predictive models linking large-scale land acquisitions to legal contestation outcomes using deliberately minimal variable sets. Model A predicts climate litigation cases from land deal activity (50 countries), while Model B predicts investor-state dispute settlement (ISDS) cases across all economic sectors (113 countries). Both models employ five predictors: number of deals, total deal size, log-transformed population, corruption perception scores, and press freedom indices.

## Model Performance Summary

The models demonstrate sharply divergent predictive capacity. Model A achieves moderate explanatory power (Train R² = 0.33, Test R² = 0.14), suggesting land deal activity captures meaningful variation in climate litigation patterns. Model B fails entirely (Train R² = 0.09, Test R² = -0.13), with negative test set performance indicating predictions worse than the sample mean. The larger sample size in Model B (113 vs 50 countries) does not compensate for fundamentally weak relationships between land acquisitions and general ISDS activity.

## Data Distribution and Zero-Inflation

Both dependent variables exhibit severe positive skew characteristic of legal case data. In the climate litigation sample, 46 of 50 countries (92%) report zero cases, with the United States appearing as an extreme outlier at 969 cases. After log transformation, the distribution remains right-skewed but approaches normality. ISDS cases show similar concentration: 63 of 113 countries (56%) report zero cases, with Venezuela (45 cases), Argentina (48 cases), and Spain (42 cases) representing the upper tail. This zero-inflation suggests legal contestation follows a highly selective process, concentrated in countries with specific institutional or political characteristics not fully captured by land deal metrics alone.

## Predictor Relationships: Model A (Climate Litigation)

The correlation matrix reveals substantial collinearity between land deal metrics. Total deal size and number of deals correlate at 0.85, indicating these measures capture overlapping information about land acquisition intensity. Governance indicators also cluster: corruption and press freedom scores correlate at 0.79, reflecting their common origin in liberal democratic institutions. This multicollinearity complicates coefficient interpretation but does not necessarily undermine prediction.

Feature importance analysis identifies number of deals as the dominant predictor (standardized coefficient: 0.54), followed by corruption score (0.40), log population (0.34), and press freedom score (0.31). Notably, total deal size exhibits a negative coefficient (-0.37) despite its positive correlation with number of deals. This reversal suggests that conditional on deal frequency, larger average deal sizes predict fewer climate cases. Countries with many small land acquisitions face more climate litigation than countries with fewer, larger deals.

The interaction term between deal count and corruption score adds minimal explanatory power (ΔR² = 0.008), indicating the relationship between land deals and climate cases does not vary systematically with governance quality in this minimal specification.

## Predictor Relationships: Model B (ISDS Cases)

Model B exhibits uniformly weak predictor coefficients. Log population emerges as the strongest variable (0.22), reflecting the mechanical tendency for larger countries to accumulate more ISDS cases over time. Press freedom score (0.19) shows modest positive association, potentially capturing institutional capacity to engage in international arbitration. Land deal variables contribute almost nothing: total deal size (0.09) and number of deals (0.07) show near-zero standardized effects.

Corruption score reverses direction relative to Model A, showing a small negative coefficient (-0.08). This suggests countries with higher corruption face marginally fewer ISDS cases, contradicting conventional assumptions about governance quality and legal exposure. The effect is too small to interpret confidently given the model's overall failure.

Correlation structures in Model B resemble Model A but with attenuated magnitudes. Deal size and deal count correlate at 0.70 rather than 0.85. Corruption and press freedom correlate at 0.69 rather than 0.79. These weaker associations suggest the 113-country sample includes more heterogeneous cases where governance and land acquisition patterns diverge.

## Residual Diagnostics

Model A residuals show moderate heteroscedasticity, with variance increasing at higher fitted values. The Q-Q plot indicates reasonable normality in the central distribution but heavy tails, particularly at the upper end where the United States outlier exerts influence. The residual histogram approximates normal distribution despite these tail deviations. Several data points exhibit large positive residuals, representing countries with more climate cases than predicted by their land acquisition profiles.

Model B residuals display more concerning patterns. The residuals-versus-fitted plot reveals systematic structure: the model overpredicts case counts for countries with low fitted values (negative residuals cluster at the left) and underpredicts for countries with high fitted values (positive residuals at the right). This funnel pattern indicates heteroscedasticity and suggests the linear specification fails to capture the data-generating process. Paradoxically, the Q-Q plot shows better normality than Model A, with residuals tracking the theoretical normal distribution closely across most quantiles. This combination—systematic prediction error alongside normally distributed residuals—typifies a model with correctly specified error structure but incorrect functional form.

## Coefficient Comparison Between Models

Direct comparison of standardized coefficients reveals fundamental differences in how land deals relate to legal contestation types. Number of deals dominates climate litigation prediction (0.54) but contributes minimally to ISDS prediction (0.07). Corruption score strongly predicts climate cases (0.40) but shows near-zero negative effect on ISDS cases (-0.08). Most strikingly, total deal size predicts fewer climate cases (-0.37) but more ISDS cases (0.09), though neither effect is strong in Model B.

These divergent patterns suggest climate litigation and ISDS represent distinct causal pathways. Climate cases appear concentrated in countries with high land acquisition activity, particularly where many deals occur in corrupt institutional environments. ISDS cases distribute more uniformly across countries regardless of land deal patterns, driven primarily by population size and general institutional capacity rather than land-specific grievances.

## Model Limitations

Both models suffer from severe sample size constraints. Model A's 10 observations per predictor falls below conventional thresholds for stable regression estimates. The 40/10 train-test split compounds this problem, with only 10 test observations providing an unreliable basis for generalization assessment. Model B's larger sample (22.6 observations per predictor) proves insufficient to overcome weak underlying relationships.

The minimal variable specification deliberately omits potentially confounding factors: treaty network structure for ISDS cases, historical litigation patterns, legal system characteristics, environmental policy stringency, and sector-specific economic factors. Corruption and press freedom scores capture governance broadly but may not reflect the specific institutional qualities relevant to legal contestation. Log population transformation assumes linear effects on the logarithmic scale, which may not hold for social processes like litigation.

Zero-inflation poses fundamental modeling challenges not addressed by standard linear regression. Countries with zero cases represent a qualitatively different category—those lacking the institutional capacity, legal frameworks, or political conditions for any litigation—rather than simply the low end of a continuous distribution. Hurdle models or zero-inflated specifications might better capture this two-stage process, first modeling the probability of any cases occurring, then modeling case counts conditional on that threshold being crossed.

## Implications for Theory and Future Analysis

The stark contrast between Model A's modest success and Model B's complete failure carries theoretical implications. Climate litigation appears more directly linked to land acquisition activity, suggesting environmental legal challenges arise from specific, observable changes in land use patterns. ISDS cases follow more diffuse causal pathways, potentially triggered by regulatory changes, contract disputes, or political instability unrelated to land deals.

The negative coefficient for total deal size in Model A presents a puzzle warranting further investigation. One interpretation: many small deals indicate agricultural investment pressure distributed across rural areas, generating localized environmental disputes that aggregate into national-level climate litigation. Fewer, larger deals may represent plantation-scale investments with different environmental footprints or community impacts. Alternatively, deal size measurement error could drive this result, as land databases struggle to accurately value transactions.

The failure to find interaction effects between land deals and governance quality suggests these factors operate additively rather than multiplicatively. Countries with high land acquisition and weak governance face litigation from both channels independently rather than experiencing compounding effects. This additive structure, if robust, simplifies prediction but complicates policy interpretation.

Future analysis should address sample size limitations through Bayesian hierarchical approaches that pool information across countries while respecting heterogeneity. Sector-specific ISDS models focusing on agriculture and mining cases (rather than all sectors) may recover the land deal signal absent in Model B. Time-series or panel specifications could exploit within-country variation over time, controlling for stable country characteristics that confound cross-sectional analysis. Most fundamentally, addressing zero-inflation through appropriate statistical models represents a prerequisite for valid inference about legal contestation patterns.

## Data Sources and Reproducibility

Analysis conducted using merged country-level data (N=113 countries) derived from Land Matrix land acquisition records, UNCTAD ISDS case database, and Sabin Center climate litigation tracker. Governance indicators drawn from Transparency International (corruption) and Reporters Without Borders (press freedom). Population data from standard international sources. All visualizations and summary statistics generated from the minimal predictor set deliberately restricted to five variables to examine baseline explanatory capacity. Complete data file: `merged_country_level_data_minimal.csv`
