# ============================================
# 1. INSTALL & LOAD LIBRARIES
install.packages("dplyr")
install.packages("caret")
install.packages("ggplot2")

library(dplyr)
library(caret)
library(ggplot2)


# 2. LOAD DATASET
df <- read.csv("C:/Users/deepu/Downloads/housing.csv")

# View data
head(df)
str(df)
summary(df)


# 3. CHECK MISSING VALUES
print(colSums(is.na(df)))


# 4. HANDLE MISSING VALUES

# Numeric columns → median
num_cols <- sapply(df, is.numeric)

for(col in names(df)[num_cols]) {
  df[[col]][is.na(df[[col]])] <- median(df[[col]], na.rm = TRUE)
}

# Categorical columns → mode
mode_func <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

cat_cols <- sapply(df, is.character)

for(col in names(df)[cat_cols]) {
  df[[col]][is.na(df[[col]])] <- mode_func(df[[col]])
}

# 5. REMOVE DUPLICATES
df <- df[!duplicated(df), ]


# 6. OUTLIER REMOVAL
df <- df[df$median_house_value < quantile(df$median_house_value, 0.99), ]


# 7. FEATURE ENGINEERING
df$rooms_per_household <- df$total_rooms / df$households
df$bedrooms_per_room <- df$total_bedrooms / df$total_rooms
df$population_per_household <- df$population / df$households


# 8. ENCODE CATEGORICAL VARIABLES
df$ocean_proximity <- as.factor(df$ocean_proximity)

dummies <- dummyVars(~ ., data = df)
df_encoded <- data.frame(predict(dummies, newdata = df))


# 9. FEATURE SCALING
preProc <- preProcess(df_encoded, method = c("center", "scale"))
df_scaled <- predict(preProc, df_encoded)


# 10. TRAIN-TEST SPLIT
set.seed(42)

trainIndex <- createDataPartition(df_scaled$median_house_value, p = 0.8, list = FALSE)

train_data <- df_scaled[trainIndex, ]
test_data  <- df_scaled[-trainIndex, ]


# 11. LINEAR REGRESSION MODEL
model <- lm(median_house_value ~ ., data = train_data)

summary(model)


# 12. PREDICTION
pred <- predict(model, newdata = test_data)


# 13. EVALUATION
rmse <- sqrt(mean((test_data$median_house_value - pred)^2))
r2 <- cor(test_data$median_house_value, pred)^2

cat("\nRMSE:", rmse)
cat("\nR² Score:", r2)


# 14. SAVE CLEAN DATA
write.csv(df_scaled, "C:/Users/deepu/OneDrive/Desktop/Machine Learning - Project/Team Tech Titans/Project Work Flow/Data Cleaning & Preprocessing/cleaned_housing_r.csv", row.names = FALSE)

cat("\n\nPROJECT COMPLETED SUCCESSFULLY")