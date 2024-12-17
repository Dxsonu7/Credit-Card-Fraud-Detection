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


print(paste("XGBoost AUC:", roc_auc_xgb))


# Install and load necessary libraries
library(randomForest)  # For Random Forest
library(yardstick)     # For evaluation metrics

# Train a Random Forest model
rf_model <- randomForest(Class ~ ., data = train_data, ntree = 100, importance = TRUE)

# Make predictions on the training and test data
rf_train_preds <- predict(rf_model, newdata = train_data)
rf_test_preds <- predict(rf_model, newdata = test_data)

# Confusion Matrix for train and test predictions
train_cm_rf <- confusionMatrix(as.factor(rf_train_preds), as.factor(y_train))
test_cm_rf <- confusionMatrix(as.factor(rf_test_preds), as.factor(y_test))

# Print the confusion matrix
print(train_cm_rf)
print(test_cm_rf)

# Calculate F1 score
f1_train_rf <- f_meas(as.factor(y_train), as.factor(rf_train_preds), positive = "1")
f1_test_rf <- f_meas(as.factor(y_test), as.factor(rf_test_preds), positive = "1")

# Print F1 scores
print(paste("Random Forest F1 Score (Train):", f1_train_rf))
print(paste("Random Forest F1 Score (Test):", f1_test_rf))
