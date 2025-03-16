# ---------------------------------
# 1. Load Required Libraries
# ---------------------------------
#install.packages(c("caret","kernlab", "tictoc"))
library(readxl)
library(car)
library(caret)
library(kernlab)  # Gaussian Process Regression
library(nnet)     # Neural Networks
library(tictoc)   # Execution time tracking

# ---------------------------------
# 2. Load and Preprocess Data
# ---------------------------------
boston840 <- read_excel("boston840.xlsx")
summary(boston840)

# Function to scale predictors to [-1, 1] except CMEDV
scalpm1 <- function(x) {
  return((x - (min(x) + max(x)) / 2) / (.5 * (max(x) - min(x))))
}

# Apply scaling to all columns except CMEDV
boston840s <- data.frame(
  CMEDV = boston840$CMEDV,  # Unscaled target variable
  sCRIM = scalpm1(boston840$CRIM),
  sINDUS = scalpm1(boston840$INDUS),
  sNOX = scalpm1(boston840$NOX),
  sRM = scalpm1(boston840$RM),
  SRM2 = scalpm1(boston840$RM)^2,  # Explicitly create squared term
  sAGE = scalpm1(boston840$AGE),
  sDIS = scalpm1(boston840$DIS),
  sTAX = scalpm1(boston840$TAX),
  sPTRATIO = scalpm1(boston840$PTRATIO),
  sLSTAT = scalpm1(boston840$LSTAT)
)

# ---------------------------------
# 3. Boxplots Before Outlier Removal
# ---------------------------------
par(mfrow = c(1, 1))  
boxplot(boston840s[2:10], horizontal = TRUE, main = "Boxplots Before Outlier Removal")

# ---------------------------------
# 4. Remove Outliers Using IQR
# ---------------------------------
remove_outliers <- function(data) {
  for (col in colnames(data)) {
    if (col != "CMEDV") {  
      Q1 <- quantile(data[[col]], 0.25, na.rm = TRUE)
      Q3 <- quantile(data[[col]], 0.75, na.rm = TRUE)
      IQR_value <- Q3 - Q1
      lower_bound <- Q1 - 1.5 * IQR_value
      upper_bound <- Q3 + 1.5 * IQR_value
      data <- data[data[[col]] >= lower_bound & data[[col]] <= upper_bound, ]
    }
  }
  return(data)
}
boston840s <- remove_outliers(boston840s)

# ---------------------------------
# 5. Boxplots After Outlier Removal
# ---------------------------------
par(mfrow = c(1, 1))  
boxplot(boston840s[2:10], horizontal = TRUE, main = "Boxplots After Outlier Removal")

# ---------------------------------
# 6. Define 10-Fold Cross-Validation
# ---------------------------------
set.seed(123)
trainMeth <- trainControl(method = "cv", number = 10)

# ---------------------------------
# 7. MULTIPLE LINEAR REGRESSION (LM) with CV
# ---------------------------------
tic("Linear Regression Model Training Time")  
CVregModel_lm <- train(
  CMEDV ~ sRM + SRM2 + sAGE + sDIS + sTAX + sPTRATIO,
  data = boston840s,
  method = "lm",
  trControl = trainMeth
)
LM_Time <- toc()
summary(CVregModel_lm)

# ---------------------------------
# 8. PREDICTIONS AND RESIDUALS
# ---------------------------------
predicted_lm <- predict(CVregModel_lm, newdata = boston840s)
resid_lm <- boston840s$CMEDV - predicted_lm

# ---------------------------------
# 9. ACTUAL VS. PREDICTED PLOT
# ---------------------------------
par(mfrow = c(1, 1))  
plot(boston840s$CMEDV, predicted_lm,
     xlab = "Actual CMEDV", 
     ylab = "Predicted CMEDV", 
     main = "Actual vs. Predicted - Linear Regression",
     col = "black")
abline(0, 1, col = "red")  # 45-degree reference line

# ---------------------------------
# 10. RESIDUALS VS. PREDICTED PLOT
# ---------------------------------
par(mfrow = c(1, 1))  
plot(predicted_lm, resid_lm,
     xlab = "Predicted CMEDV", 
     ylab = "Residuals", 
     main = "Residuals vs. Predicted - Linear Regression",
     col = "blue")
abline(h = 0, col = "red")  # Reference line at zero

# ---------------------------------
# 11. MODEL QUALITY CHECKS
# ---------------------------------
# (A) Autocorrelation Check for Residuals
par(mfrow = c(1, 1))  
acf(resid_lm, main = "Autocorrelation of Residuals - Linear Regression")

# (B) Normality Check with Q-Q Plot
par(mfrow = c(1, 1))  
qqnorm(resid_lm, main = "Q-Q Plot - Linear Regression Residuals")
qqline(resid_lm, col = "red")  # Add reference line

# (C) Residuals vs. Each Predictor
par(mfrow = c(2, 3))  
plot(boston840s$sRM, resid_lm, main = "Residuals vs sRM", col = "blue")
plot(boston840s$SRM2, resid_lm, main = "Residuals vs SRM2", col = "blue")
plot(boston840s$sAGE, resid_lm, main = "Residuals vs sAGE", col = "blue")
plot(boston840s$sDIS, resid_lm, main = "Residuals vs sDIS", col = "blue")
plot(boston840s$sTAX, resid_lm, main = "Residuals vs sTAX", col = "blue")
plot(boston840s$sPTRATIO, resid_lm, main = "Residuals vs sPTRATIO", col = "blue")
abline(h = 0, col = "red")  

# ---------------------------------
# 12. FINAL MODEL SUMMARY
# ---------------------------------
print("Model Summary for Linear Regression with Cross-Validation:")
print(CVregModel_lm)

# Extract Cross-Validation Results
print("Cross-Validation Steps Summary:")
print(CVregModel_lm$resample)

# Extract Regression Coefficients
print("Coefficients from Final Model:")
print(CVregModel_lm$finalModel$coefficients)

# Extract Model Performance Metrics
print("Cross-Validation Model Results:")
print(CVregModel_lm$results)

# Compute RMSE (Sigma)
print(paste("Model RMSE (Sigma):", CVregModel_lm$sigma))

# ---------------------------------
# 13. RMSE AND R-SQUARED UNCERTAINTY ASSESSMENT
# ---------------------------------
print("Uncertainty Assessment:")
print(c("RMSE", CVregModel_lm$results$RMSE))
print(c("Standard Deviation of RMSE", CVregModel_lm$results$RMSESD))
print(c("R-squared", CVregModel_lm$results$Rsquared))
print(c("Standard Deviation of R-squared", CVregModel_lm$results$RsquaredSD))

# ---------------------------------
# 14. VARIABLE IMPORTANCE PLOT
# ---------------------------------
plot(varImp(CVregModel_lm), main = "Variable Importance - Linear Regression")

# ---------------------------------
# 15. COMPUTE PREDICTION INTERVALS
# ---------------------------------
predictVals <- predict(CVregModel_lm, newdata = boston840s)

# Compute approximate confidence intervals for predictions
pUpper <- predictVals + 2 * CVregModel_lm$results$RMSE
pLower <- predictVals - 2 * CVregModel_lm$results$RMSE

# Print approximate prediction intervals
print("Approximate Prediction Intervals:")
print(cbind(Predicted = predictVals, Lower = pLower, Upper = pUpper))


# ---------------------------------
# 16. GAUSSIAN PROCESS REGRESSION (GP) MODEL
# ---------------------------------

set.seed(123)
tic("Gaussian Process Regression Model Training Time")
CVregModel_gp <- train(
  CMEDV ~ sRM + SRM2 + sAGE + sDIS + sTAX + sPTRATIO,
  data = boston840s,
  method = "gaussprRadial",   # Gaussian Process with radial (exponential) kernel
  trControl = trainMeth,
  preProc = c("center", "scale")  # Optional pre-processing
)
GP_Time <- toc()

print("Model Summary for Gaussian Process Regression (gaussprRadial):")
print(CVregModel_gp)

# ---------------------------------
# 17. VARIABLE IMPORTANCE PLOT for GP MODEL
# ---------------------------------
plot(varImp(CVregModel_gp), main = "Variable Importance - GP Regression (gaussprRadial)")


# ---------------------------------
# 18. PRACTICAL PREDICTIVE ABILITY: FITTED LINE PLOT, RMSE, and R-SQUARED
# ---------------------------------
predicted_gp <- predict(CVregModel_gp, newdata = boston840s)
resid_gp <- boston840s$CMEDV - predicted_gp

# Fitted line plot: Actual vs. Predicted for GP model
par(mfrow = c(1, 1))
plot(boston840s$CMEDV, predicted_gp,
     xlab = "Actual CMEDV", 
     ylab = "Predicted CMEDV", 
     main = "Actual vs. Predicted - GP Regression",
     col = "black")
abline(0, 1, col = "red")  # 45-degree reference line

# Extract and print RMSE and R-squared for GP model
gp_RMSE <- CVregModel_gp$results$RMSE
gp_Rsquared <- CVregModel_gp$results$Rsquared
print(paste("GP Model RMSE:", gp_RMSE))
print(paste("GP Model R-squared:", gp_Rsquared))


# ---------------------------------
# 19. PREDICTION UNCERTAINTY: RMSESD and R-SQUARED SD for GP MODEL
# ---------------------------------
gp_RMSESD <- CVregModel_gp$results$RMSESD
gp_RsquaredSD <- CVregModel_gp$results$RsquaredSD
print("GP Model Uncertainty Assessment:")
print(c("RMSESD", gp_RMSESD))
print(c("R-squaredSD", gp_RsquaredSD))


# ---------------------------------
# 20. MODEL QUALITY CHECKS for GP MODEL
# ---------------------------------
# (A) Autocorrelation of residuals
par(mfrow = c(1, 1))
acf(resid_gp, main = "Autocorrelation of Residuals - GP Regression")


# (B) Normality check using Q-Q Plot
par(mfrow = c(1, 1))
qqnorm(resid_gp, main = "Q-Q Plot - GP Regression Residuals")
qqline(resid_gp, col = "red")


# (C) Residuals vs. Predictors
par(mfrow = c(2, 3))
plot(boston840s$sRM, resid_gp, main = "Residuals vs sRM (GP)", col = "blue")
plot(boston840s$SRM2, resid_gp, main = "Residuals vs SRM2 (GP)", col = "blue")
plot(boston840s$sAGE, resid_gp, main = "Residuals vs sAGE (GP)", col = "blue")
plot(boston840s$sDIS, resid_gp, main = "Residuals vs sDIS (GP)", col = "blue")
plot(boston840s$sTAX, resid_gp, main = "Residuals vs sTAX (GP)", col = "blue")
plot(boston840s$sPTRATIO, resid_gp, main = "Residuals vs sPTRATIO (GP)", col = "blue")
abline(h = 0, col = "red")


# ---------------------------------
# 21. COMPUTE PREDICTION VALUES AND CV-BASED PREDICTION INTERVALS
#     for TYPICAL and HIGH VALUE NEIGHBORHOODS using the GP model
# ---------------------------------
# Define a "typical" neighborhood using the median of predictors
typical <- data.frame(
  sRM = median(boston840s$sRM, na.rm = TRUE),
  SRM2 = median(boston840s$SRM2, na.rm = TRUE),
  sAGE = median(boston840s$sAGE, na.rm = TRUE),
  sDIS = median(boston840s$sDIS, na.rm = TRUE),
  sTAX = median(boston840s$sTAX, na.rm = TRUE),
  sPTRATIO = median(boston840s$sPTRATIO, na.rm = TRUE)
)

# Define a "high value" neighborhood as the one with the highest CMEDV
high_value <- boston840s[which.max(boston840s$CMEDV), c("sRM", "SRM2", "sAGE", "sDIS", "sTAX", "sPTRATIO")]

# Prediction and approximate CV-based prediction intervals (using ±2*RMSE) for typical neighborhood
pred_typical <- predict(CVregModel_gp, newdata = typical)
CI_typical_upper <- pred_typical + 2 * gp_RMSE
CI_typical_lower <- pred_typical - 2 * gp_RMSE

# Prediction and intervals for high value neighborhood
pred_high <- predict(CVregModel_gp, newdata = high_value)
CI_high_upper <- pred_high + 2 * gp_RMSE
CI_high_lower <- pred_high - 2 * gp_RMSE

print("Prediction for Typical Neighborhood (GP Regression):")
print(data.frame(Predicted = pred_typical, Lower = CI_typical_lower, Upper = CI_typical_upper))

print("Prediction for High Value Neighborhood (GP Regression):")
print(data.frame(Predicted = pred_high, Lower = CI_high_lower, Upper = CI_high_upper))

# ---------------------------------
# 22. NEURAL NETWORK REGRESSION (NNET) MODEL
# ---------------------------------
set.seed(123) 

tic("Neural Network Regression Model Training Time")
CVregModel_nnet <- train(
  CMEDV ~ sRM + SRM2 + sAGE + sDIS + sTAX + sPTRATIO,
  data = boston840s,
  method = "nnet",           # Neural Network Regression
  trControl = trainMeth,
  linout = TRUE,             # Regression output
  trace = FALSE,             # Suppress detailed training output
  tuneGrid = expand.grid(
    size = c(5, 9, 9),       # Number of hidden units
    decay = c(0, 0.01, 0.1)  # Regularization parameter
  )
)
NNET_Time <- toc()


print("Model Summary for Neural Network Regression (nnet):")
print(CVregModel_nnet)


# ---------------------------------
# 23. VARIABLE IMPORTANCE PLOT for NNET MODEL
# ---------------------------------
plot(varImp(CVregModel_nnet), main = "Variable Importance - Neural Network Regression (nnet)")

# ---------------------------------
# 24. PRACTICAL PREDICTIVE ABILITY: FITTED LINE PLOT, RMSE, and R-SQUARED
# ---------------------------------
predicted_nnet <- predict(CVregModel_nnet, newdata = boston840s)
resid_nnet <- boston840s$CMEDV - predicted_nnet

# Fitted line plot: Actual vs. Predicted for NNET model
par(mfrow = c(1, 1))
plot(boston840s$CMEDV, predicted_nnet,
     xlab = "Actual CMEDV", 
     ylab = "Predicted CMEDV", 
     main = "Actual vs. Predicted - NNET Regression",
     col = "black")
abline(0, 1, col = "red")

# Extract and print RMSE and R-squared for NNET model
nnet_RMSE <- CVregModel_nnet$results$RMSE
nnet_Rsquared <- CVregModel_nnet$results$Rsquared
print(paste("NNET Model RMSE:", nnet_RMSE))
print(paste("NNET Model R-squared:", nnet_Rsquared))


# ---------------------------------
# 25. PREDICTION UNCERTAINTY: RMSESD and R-SQUARED SD for NNET MODEL
# ---------------------------------
nnet_RMSESD <- CVregModel_nnet$results$RMSESD
nnet_RsquaredSD <- CVregModel_nnet$results$RsquaredSD
print("NNET Model Uncertainty Assessment:")
print(c("RMSESD", nnet_RMSESD))
print(c("R-squaredSD", nnet_RsquaredSD))


# ---------------------------------
# 26. MODEL QUALITY CHECKS for NNET MODEL
# ---------------------------------
# (A) Autocorrelation of residuals
par(mfrow = c(1, 1))
acf(resid_nnet, main = "Autocorrelation of Residuals - NNET Regression")


# (B) Normality check using Q-Q Plot
par(mfrow = c(1, 1))
qqnorm(resid_nnet, main = "Q-Q Plot - NNET Regression Residuals")
qqline(resid_nnet, col = "red")


# (C) Residuals vs. Predictors
par(mfrow = c(2, 3))
plot(boston840s$sRM, resid_nnet, main = "Residuals vs sRM (NNET)", col = "blue")
plot(boston840s$SRM2, resid_nnet, main = "Residuals vs SRM2 (NNET)", col = "blue")
plot(boston840s$sAGE, resid_nnet, main = "Residuals vs sAGE (NNET)", col = "blue")
plot(boston840s$sDIS, resid_nnet, main = "Residuals vs sDIS (NNET)", col = "blue")
plot(boston840s$sTAX, resid_nnet, main = "Residuals vs sTAX (NNET)", col = "blue")
plot(boston840s$sPTRATIO, resid_nnet, main = "Residuals vs sPTRATIO (NNET)", col = "blue")
abline(h = 0, col = "red")


# ---------------------------------
# 27. COMPUTE PREDICTION VALUES AND CV-BASED PREDICTION INTERVALS
#     for TYPICAL and HIGH VALUE NEIGHBORHOODS using the NNET model
# ---------------------------------
# Prediction and approximate CV-based prediction intervals (using ±2*RMSE) for typical neighborhood
pred_typical_nnet <- predict(CVregModel_nnet, newdata = typical)
CI_typical_upper_nnet <- pred_typical_nnet + 2 * nnet_RMSE
CI_typical_lower_nnet <- pred_typical_nnet - 2 * nnet_RMSE

# Prediction and intervals for high value neighborhood
pred_high_nnet <- predict(CVregModel_nnet, newdata = high_value)
CI_high_upper_nnet <- pred_high_nnet + 2 * nnet_RMSE
CI_high_lower_nnet <- pred_high_nnet - 2 * nnet_RMSE

print("Prediction for Typical Neighborhood (NNET Regression):")
print(data.frame(Predicted = pred_typical_nnet, Lower = CI_typical_lower_nnet, Upper = CI_typical_upper_nnet))

print("Prediction for High Value Neighborhood (NNET Regression):")
print(data.frame(Predicted = pred_high_nnet, Lower = CI_high_lower_nnet, Upper = CI_high_upper_nnet))


# ---------------------------------
# 28. SUMMARY OF ALL MODELS (LM, GP, NNET)
# ---------------------------------
# Extract the best row (with the lowest RMSE) for each model
lm_best   <- CVregModel_lm$results[which.min(CVregModel_lm$results$RMSE), ]
gp_best   <- CVregModel_gp$results[which.min(CVregModel_gp$results$RMSE), ]
nnet_best <- CVregModel_nnet$results[which.min(CVregModel_nnet$results$RMSE), ]

# Now create the summary table using the best performance metrics
summary_results <- data.frame(
  Model = c("Linear Regression", "Gaussian Process Regression", "Neural Network Regression"),
  RMSE = c(lm_best$RMSE, gp_best$RMSE, nnet_best$RMSE),
  R_squared = c(lm_best$Rsquared, gp_best$Rsquared, nnet_best$Rsquared),
  RMSE_SD = c(lm_best$RMSESD, gp_best$RMSESD, nnet_best$RMSESD),
  R_squared_SD = c(lm_best$RsquaredSD, gp_best$RsquaredSD, nnet_best$RsquaredSD)
)

print("Summary of Model Performance:")
print(summary_results)


# ---------------------------------
# 28. SUMMARY OF ALL MODELS (LM, GP, NNET)
# ---------------------------------

# Extract the best row (with the lowest RMSE) for each model
lm_best   <- CVregModel_lm$results[which.min(CVregModel_lm$results$RMSE), ]
gp_best   <- CVregModel_gp$results[which.min(CVregModel_gp$results$RMSE), ]
nnet_best <- CVregModel_nnet$results[which.min(CVregModel_nnet$results$RMSE), ]

# Compute predictions for 'typical' and 'high-value' neighborhoods
pred_typical_lm <- predict(CVregModel_lm, newdata = typical)
pred_high_lm    <- predict(CVregModel_lm, newdata = high_value)

pred_typical_gp <- predict(CVregModel_gp, newdata = typical)
pred_high_gp    <- predict(CVregModel_gp, newdata = high_value)

pred_typical_nnet <- predict(CVregModel_nnet, newdata = typical)
pred_high_nnet    <- predict(CVregModel_nnet, newdata = high_value)

# Compute Prediction Intervals (± 2 * RMSE)
CI_typical_lm <- c(pred_typical_lm - 2 * lm_best$RMSE, pred_typical_lm + 2 * lm_best$RMSE)
CI_high_lm    <- c(pred_high_lm - 2 * lm_best$RMSE, pred_high_lm + 2 * lm_best$RMSE)

CI_typical_gp <- c(pred_typical_gp - 2 * gp_best$RMSE, pred_typical_gp + 2 * gp_best$RMSE)
CI_high_gp    <- c(pred_high_gp - 2 * gp_best$RMSE, pred_high_gp + 2 * gp_best$RMSE)

CI_typical_nnet <- c(pred_typical_nnet - 2 * nnet_best$RMSE, pred_typical_nnet + 2 * nnet_best$RMSE)
CI_high_nnet    <- c(pred_high_nnet - 2 * nnet_best$RMSE, pred_high_nnet + 2 * nnet_best$RMSE)

# Create the summary table with model performance metrics
summary_results <- data.frame(
  Model = c("Linear Regression", "Gaussian Process Regression", "Neural Network Regression"),
  RMSE = c(lm_best$RMSE, gp_best$RMSE, nnet_best$RMSE),
  R_squared = c(lm_best$Rsquared, gp_best$Rsquared, nnet_best$Rsquared),
  RMSE_SD = c(lm_best$RMSESD, gp_best$RMSESD, nnet_best$RMSESD),
  R_squared_SD = c(lm_best$RsquaredSD, gp_best$RsquaredSD, nnet_best$RsquaredSD),
  Fitting_Time = c(paste(round(LM_Time$toc - LM_Time$tic, 2), "sec"), paste(round(GP_Time$toc - GP_Time$tic, 2), "sec"), paste(round(NNET_Time$toc - NNET_Time$tic, 2), "sec"))  # Update actual times if stored
)

# Create predictions summary table
prediction_summary <- data.frame(
  Model = rep(c("Linear Regression", "Gaussian Process Regression", "Neural Network Regression"), each = 2),
  Neighborhood = rep(c("Typical", "High-Value"), times = 3),
  Predicted = c(pred_typical_lm, pred_high_lm, pred_typical_gp, pred_high_gp, pred_typical_nnet, pred_high_nnet),
  Lower_CI = c(CI_typical_lm[1], CI_high_lm[1], CI_typical_gp[1], CI_high_gp[1], CI_typical_nnet[1], CI_high_nnet[1]),
  Upper_CI = c(CI_typical_lm[2], CI_high_lm[2], CI_typical_gp[2], CI_high_gp[2], CI_typical_nnet[2], CI_high_nnet[2])
)

# Print model performance summary
print("Summary of Model Performance:")
print(summary_results)

# Print predictions with intervals
print("Prediction Summary for Typical and High-Value Neighborhoods:")
print(prediction_summary)









