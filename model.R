## Load Libaries

library(tidyverse)       # Data manipulation and visualization
library(caret)           # Train-test split, preprocessing
library(recipes)         # Preprocessing steps
library(xgboost)         # XGBoost model
library(keras)           # Artificial Neural Network
library(yardstick)       # Model evaluation (AUC, confusion matrix)

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

