## Load Libaries

library(tidyverse)       # Data manipulation and visualization
library(caret)           # Train-test split, preprocessing
library(recipes)         # Preprocessing steps
library(xgboost)         # XGBoost model
library(keras)           # Artificial Neural Network
library(yardstick)       # Model evaluation (AUC, confusion matrix)
library(corrplot)

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


