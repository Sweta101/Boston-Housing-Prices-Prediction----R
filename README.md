# Boston Housing Price Prediction Models

This repository explores predictive modeling of Boston housing prices using a data-adaptive framework, comparing Multiple Linear Regression, Gaussian Process Regression, and Neural Network Regression, all evaluated with 10-fold cross-validation.

## Overview

Based on a corrected version of the 1970 Boston Housing dataset (506 town tracts), the goal is to develop robust predictive models that accurately forecast median home values (CMEDV) using community characteristics. The analysis aids decision-making in urban planning, real estate investment, and policy formulation.

## Models Compared

- **Multiple Linear Regression (LM):** Traditional regression approach; fast and interpretable.
- **Gaussian Process Regression (GPR):** Offers stable predictions with uncertainty quantification.
- **Neural Network Regression (NNR):** Most accurate; captures nonlinear relationships at the cost of higher computational time.

## Methodology

- Data preprocessing: Scaling to [–1, 1], outlier removal using IQR method.
- Evaluation metrics: RMSE, R², MAE, prediction intervals.
- Cross-validation: 10-fold CV to assess prediction error robustness.
- Visualization: Actual vs. Predicted plots, Residuals analysis, Variable importance charts.

## Key Insights

- NNR had the highest predictive accuracy (R² ≈ 0.75, RMSE ≈ $2.10k).
- GPR provided the most consistent results with low standard deviations in performance.
- LM was computationally efficient and a solid baseline model.
- Model selection depends on use-case: speed (LM), consistency (GPR), or accuracy (NNR).

## Usage

All models and visualizations were developed in R using packages like `caret`, `readxl`, and base plotting tools. Scripts are structured to allow easy reproduction and extension.

## License

MIT License

---

*This project demonstrates how machine learning models can be leveraged to provide actionable insights in housing markets by quantifying prediction uncertainty and enabling better decision-making.*
