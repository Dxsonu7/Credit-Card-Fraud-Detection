## Load Libaries
library(tidyverse)       # Data manipulation and visualization
library(caret)           # Train-test split, preprocessing
library(recipes)         # Preprocessing steps
library(xgboost)         # XGBoost model
library(keras)           # Artificial Neural Network
library(yardstick)       # Model evaluation (AUC, confusion matrix)
library(corrplot)
library(reticulate)


## Read data and define research
data <- read_csv("creditcard.csv")

# Basic structure, dimension and summary
head(data)
dim(data)
glimpse(data)
summary(data)


# Check for class imbalance
data %>%
  count(Class) %>%
  mutate(percentage = n / sum(n) * 100) 

  
# Visualize class imbalance
data %>%
  ggplot(aes(x = factor(Class))) +
  geom_bar(fill = "steelblue") +
  labs(title = "Class Distribution", x = "Class", y = "Count")


# Check for missing values
colSums(is.na(data))


# Summary of numeric features (central tendency)
data %>%
  select(-Class) %>%
  summary()


# Boxplot for outliers in a few variables
data %>%
  gather(key = "feature", value = "value", V1:V5) %>%
  ggplot(aes(x = feature, y = value)) +
  geom_boxplot(fill = "lightblue") +
  labs(title = "Boxplots of Features")


# Correlation Heatmap
cor_matrix <- cor(data %>% select(-Class))
corrplot(cor_matrix, method = "color", type = "upper", tl.cex = 0.6)


# Train-test split
set.seed(42)
train_index <- createDataPartition(data$Class, p = 0.8, list = FALSE)

train_data <- data[train_index, ]
test_data <- data[-train_index, ]


# Define preprocessing recipe
rec <- recipe(Class ~ ., data = train_data) %>%
  step_center(all_predictors()) %>%
  step_scale(all_predictors()) %>%
  prep()


# Apply scaling
X_train <- bake(rec, new_data = train_data) %>% select(-Class)
y_train <- train_data$Class

X_test <- bake(rec, new_data = test_data) %>% select(-Class)
y_test <- test_data$Class


# Convert to xgb.DMatrix
dtrain <- xgb.DMatrix(data = as.matrix(X_train), label = y_train)
dtest <- xgb.DMatrix(data = as.matrix(X_test), label = y_test)


# XGBoost parameters
params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  max_depth = 6,
  eta = 0.1,
  subsample = 0.8,
  colsample_bytree = 0.8
)


# Train the model
xgb_model <- xgb.train(params = params, data = dtrain, nrounds = 100,
                       watchlist = list(train = dtrain), verbose = 0)


# Predict probabilities
xgb_preds <- predict(xgb_model, as.matrix(X_test))
xgb_class <- ifelse(xgb_preds > 0.5, 1, 0)

# Confusion Matrix
confusion_matrix_xgb <- confusionMatrix(as.factor(xgb_class), as.factor(y_test))
print(confusion_matrix_xgb)

# AUC
roc_auc_xgb <- roc_auc_vec(as.factor(y_test), xgb_preds)
print(paste("XGBoost AUC:", roc_auc_xgb))



# Hyperparameter tuning (if needed)
params_tune <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  max_depth = 6,
  eta = 0.1,
  subsample = 0.8,
  colsample_bytree = 0.8,
  scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)  # Adjust for class imbalance
)

# Perform cross-validation
cv_results <- xgb.cv(params = params_tune, data = dtrain, nrounds = 100, 
                     nfold = 5, showsd = TRUE, stratified = TRUE, 
                     print_every_n = 10, early_stopping_rounds = 10)

# Train the final model with the best nrounds from cross-validation
best_nrounds <- cv_results$best_iteration
xgb_model_tuned <- xgb.train(params = params_tune, data = dtrain, nrounds = best_nrounds,
                             watchlist = list(train = dtrain), verbose = 0)

# Make predictions and evaluate
xgb_preds_tuned <- predict(xgb_model_tuned, as.matrix(X_test))
xgb_class_tuned <- ifelse(xgb_preds_tuned > 0.5, 1, 0)
confusion_matrix_xgb_tuned <- confusionMatrix(as.factor(xgb_class_tuned), as.factor(y_test))
print(confusion_matrix_xgb_tuned)

roc_auc_xgb_tuned <- roc_auc_vec(as.factor(y_test), xgb_preds_tuned)
print(paste("Tuned XGBoost AUC:", roc_auc_xgb_tuned))

f1 <- f_meas(data = tibble(truth = as.factor(y_test), estimate = as.factor(xgb_class_tuned)), 
             truth = "truth", estimate = "estimate")
print(f1)


# Load the ROSE package
library(ROSE)

# Apply ROSE to balance the classes
train_data_rose <- ROSE(Class ~ ., data = train_data, seed = 42)$data

# Separate features and target variable
X_train_rose <- train_data_rose %>% select(-Class)
y_train_rose <- train_data_rose$Class

# Train an XGBoost model on the balanced data
dtrain_rose <- xgb.DMatrix(data = as.matrix(X_train_rose), label = y_train_rose)
dtest <- xgb.DMatrix(data = as.matrix(X_test), label = y_test)

# Train the model
xgb_model_rose <- xgb.train(params = params, data = dtrain_rose, nrounds = 100,
                            watchlist = list(train = dtrain_rose), verbose = 0)

# Make predictions and evaluate
xgb_preds_rose <- predict(xgb_model_rose, as.matrix(X_test))
xgb_class_rose <- ifelse(xgb_preds_rose > 0.5, 1, 0)
confusion_matrix_xgb_rose <- confusionMatrix(as.factor(xgb_class_rose), as.factor(y_test))
print(confusion_matrix_xgb_rose)

roc_auc_xgb_rose <- roc_auc_vec(as.factor(y_test), xgb_preds_rose)
print(paste("ROSE XGBoost AUC:", roc_auc_xgb_rose))

