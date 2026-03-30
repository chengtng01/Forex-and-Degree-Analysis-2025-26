# 1. Load the dataset
# Note: header=FALSE might be needed if your file has no column names (common in forex data)
# But let's try with header=TRUE first.
forex_data <- read.csv("eurusd.csv", header = TRUE, stringsAsFactors = FALSE)

# 2. Check the structure
str(forex_data)

# 3. Peek at the first few rows
head(forex_data)

# 1. Select the "Big 3" Indicators + Price Data
# We are choosing RSI (Momentum), MACD (Trend), and Bollinger Band Width (Volatility)
# These are the standard tools for any "Forex Sniper"
clean_forex <- forex_data[, c("Date", "Open", "High", "Low", "Close", 
                              "rsi14_H1", "macd_H1", "volatility_bbw_ta_H1")]

# 2. Rename columns to be friendlier
colnames(clean_forex) <- c("Date", "Open", "High", "Low", "Close", "RSI", "MACD", "BB_Width")

# --- CREATING THE TARGETS ---

# 3. Create the Targets (Current Candle)
# Regression Target: How "tall" is the candle? (High - Low)
clean_forex$Range <- clean_forex$High - clean_forex$Low

# Classification Target: Is it Green (1) or Red (0)?
# If Close > Open, it's a Bull candle (1).
clean_forex$Direction <- ifelse(clean_forex$Close > clean_forex$Open, 1, 0)

# 4. The "Time Shift" (CRITICAL STEP)
# We want to use Row 1 to predict Row 2.
# So we create "Target_Range" which is the Range of the *next* hour.
# We shift the vector up by 1.
clean_forex$Target_Next_Range <- c(clean_forex$Range[2:nrow(clean_forex)], NA)
clean_forex$Target_Next_Direction <- c(clean_forex$Direction[2:nrow(clean_forex)], NA)

# 5. Remove the very last row (because it has no "Next Hour" to predict)
clean_forex <- na.omit(clean_forex)

# 6. Check the final clean table
head(clean_forex)


# 1. Remove duplicate rows based on the 'Date' column
# We keep only the first instance of every timestamp
clean_forex <- clean_forex[!duplicated(clean_forex$Date), ]

# 2. Reset the index (just housekeeping)
rownames(clean_forex) <- NULL

# 3. Check again to make sure dates are now different
head(clean_forex)


# 1. Set Seed for Reproducibility
set.seed(123)

# 2. Determine the split point (70% Training, 30% Testing)
# Since this is time-series data, we CANNOT shuffle it randomly.
# We must slice it: First 70% for training, Last 30% for testing.
cutoff_index <- floor(0.7 * nrow(clean_forex))

# 3. Create the sets
train_data <- clean_forex[1:cutoff_index, ]
test_data  <- clean_forex[(cutoff_index + 1):nrow(clean_forex), ]

# 4. Check the sizes
print(paste("Training Rows:", nrow(train_data)))
print(paste("Testing Rows:", nrow(test_data)))


# 1. Train the Linear Model (lm)
# Formula: Target ~ Predictors
model_linear <- lm(Target_Next_Range ~ RSI + MACD + BB_Width + Range, 
                   data = train_data)

# 2. View the Summary (The Coefficients)
# This tells us WHICH variable is mathematically significant
summary(model_linear)



# 1. Load the Library
# (If this fails, run: install.packages("rpart") in your console)
library(rpart)

# 2. Train the Regression Tree
# method="anova" tells R we are predicting a number (Regression), not a class.
model_tree <- rpart(Target_Next_Range ~ RSI + MACD + BB_Width + Range, 
                    data = train_data, 
                    method = "anova")

# 3. Visualizing the "Brain" of the AI
# This draws the decision rules the computer learned
plot(model_tree, margin = 0.1)
text(model_tree, use.n = TRUE, cex = 0.8)



# 1. Make Predictions on the "Unseen" Test Data
preds_linear <- predict(model_linear, test_data)
preds_tree   <- predict(model_tree, test_data)

# 2. Calculate the RMSE (The Scorecard)
# Formula: Sqrt(Average( (Actual - Predicted)^2 ))
rmse_linear <- sqrt(mean((test_data$Target_Next_Range - preds_linear)^2))
rmse_tree   <- sqrt(mean((test_data$Target_Next_Range - preds_tree)^2))

# 3. Print the Results
print(paste("Linear Regression Error:", round(rmse_linear, 6)))
print(paste("Regression Tree Error:  ", round(rmse_tree, 6)))


# 1. Convert the Target to a "Factor" (Category)
# This is crucial. Without this, R will try to do regression again.
train_data$Target_Next_Direction <- as.factor(train_data$Target_Next_Direction)
test_data$Target_Next_Direction  <- as.factor(test_data$Target_Next_Direction)

# --- MODEL A: LOGISTIC REGRESSION ---
# family="binomial" tells R to do Logistic Regression (Yes/No)
model_logit <- glm(Target_Next_Direction ~ RSI + MACD + BB_Width + Range, 
                   data = train_data, 
                   family = "binomial")

# --- MODEL B: CLASSIFICATION TREE ---
# method="class" tells R to vote on the Category
model_class_tree <- rpart(Target_Next_Direction ~ RSI + MACD + BB_Width + Range, 
                          data = train_data, 
                          method = "class")

# 3. Confirmation
print("Both Classification Models Trained Successfully.")



# --- 1. Evaluate Logistic Regression ---
# Get probabilities (e.g., 0.55, 0.42)
probs_logit <- predict(model_logit, test_data, type = "response")

# Convert probabilities to Class (0 or 1) using 0.5 as the cutoff
preds_logit_class <- ifelse(probs_logit > 0.5, 1, 0)

# Calculate Accuracy (Correct Guesses / Total Guesses)
acc_logit <- mean(preds_logit_class == test_data$Target_Next_Direction)


# --- 2. Evaluate Classification Tree ---
# Get the class directly (type="class")
preds_tree_class <- predict(model_class_tree, test_data, type = "class")

# Calculate Accuracy
acc_tree <- mean(preds_tree_class == test_data$Target_Next_Direction)


# --- 3. The Scoreboard ---
print(paste("Logistic Regression Accuracy:", round(acc_logit * 100, 2), "%"))
print(paste("Classification Tree Accuracy:", round(acc_tree * 100, 2), "%"))

install.packages("rpart.plot")

# Load library for nice tree plotting
# If this fails, run: install.packages("rpart.plot")
library(rpart.plot)

# Visualize the Winning Model (Classification Tree)
# This will draw a nice flowchart showing exactly how it decides "Buy" vs "Sell"
rpart.plot(model_class_tree, 
           type = 4, 
           extra = 104, 
           tweak = 1.2, 
           box.palette = "GnBu",
           main = "The Forex Sniper Strategy: Decision Tree Rules")



# 1. Force the Tree to Grow
# We set 'cp = 0.001' (it lowers the bar for what counts as a 'good rule')
model_tree_forced <- rpart(Target_Next_Direction ~ RSI + MACD + BB_Width + Range, 
                           data = train_data, 
                           method = "class",
                           control = rpart.control(cp = 0.001, minsplit = 10))

# 2. Plot the New Detailed Tree
rpart.plot(model_tree_forced, 
           type = 4, 
           extra = 104, 
           tweak = 1.0, # Adjusted tweak to make it fit
           box.palette = "GnBu",
           main = "The 'Forced' Forex Sniper Strategy")




# 1. Prune the tree (Cut off the weak branches)
# We set cp to 0.005 (a middle ground between 0.01 and 0.001)
model_pruned <- prune(model_tree_forced, cp = 0.0025)

# 2. Plot the "Clean" Tree
rpart.plot(model_pruned, 
           type = 4, 
           extra = 104, 
           tweak = 1.0, 
           box.palette = "GnBu",
           main = "The 'Forex Sniper' Logic (Top Rules)")




# 1. Prune Aggressively
# Increasing cp from 0.0025 to 0.005 will cut off many more branches
model_final <- prune(model_tree_forced, cp = 0.005)

# 2. Plot the "Executive Summary" Tree
rpart.plot(model_final, 
           type = 4, 
           extra = 104, 
           tweak = 1.2, 
           box.palette = "GnBu",
           main = "Forex Sniper: The Top 3 Rules")

# 3. Print the Rules as Text (Backup)
# If the plot is still hard to read, this prints the logic clearly in the console
print(model_final)



# 1. Aggressive Pruning (cp = 0.015)
# This is a very high bar. It will strip the tree down to its bare bones.
model_executive <- prune(model_tree_forced, cp = 0.015)

# 2. Plot the "Executive" Tree
rpart.plot(model_executive, 
           type = 4, 
           extra = 104, 
           tweak = 1.2, 
           box.palette = "GnBu", 
           main = "The 'Forex Sniper' Core Logic")

# 3. Print the text rules (Backup validation)
print(model_executive)



# 1. Train with a "Depth Limit" (maxdepth = 3)
# We set cp low (0.001) to encourage splitting, but maxdepth=3 to stop it from getting messy.
model_report <- rpart(Target_Next_Direction ~ RSI + MACD + BB_Width + Range, 
                      data = train_data, 
                      method = "class",
                      control = rpart.control(cp = 0.001, maxdepth = 3))

# 2. Plot the "Perfect" Tree
rpart.plot(model_report, 
           type = 4, 
           extra = 104, 
           tweak = 1.2, 
           box.palette = "GnBu", 
           main = "Forex Sniper: The Top 3 Decision Rules")



# 1. Create the Confusion Matrix
# We compare the "Tree's Guess" (preds_tree_class) vs "Reality" (test_data$Target_Next_Direction)
conf_matrix <- table(Predicted = preds_tree_class, Actual = test_data$Target_Next_Direction)

# 2. Print it
print(conf_matrix)

# 3. Calculate "Precision" (Win Rate when it actually trades)
# Formula: True Positives / (True Positives + False Positives)
# Note: We assume '1' is the second column/row. 
true_positives <- conf_matrix[2, 2]
false_positives <- conf_matrix[2, 1]
precision <- true_positives / (true_positives + false_positives)

print(paste("Precision (Win Rate of Buy Signals):", round(precision * 100, 2), "%"))



# 1. Make predictions using the BETTER model (model_report)
# This model has the "Buy" rule (Blue Box) inside it
preds_new <- predict(model_report, test_data, type = "class")

# 2. Create the new Confusion Matrix
conf_matrix_new <- table(Predicted = preds_new, Actual = test_data$Target_Next_Direction)
print(conf_matrix_new)

# 3. Calculate Precision again
# We check row "1" (Predicted Buy)
# Note: If the matrix doesn't have a row "1", it means it STILL didn't buy.
if("1" %in% rownames(conf_matrix_new)) {
  true_positives <- conf_matrix_new["1", "1"]
  false_positives <- conf_matrix_new["1", "0"]
  precision <- true_positives / (true_positives + false_positives)
  print(paste("Precision (Win Rate of Buy Signals):", round(precision * 100, 2), "%"))
} else {
  print("The model is still too conservative! It found 0 Buy signals in the Test Data.")
}

install.packages("pROC")

# 1. Load Library
library(pROC)

# 2. Calculate ROC for Logistic Regression (The Baseline)
# We use the probabilities we calculated earlier
roc_logit <- roc(test_data$Target_Next_Direction, probs_logit)

# 3. Calculate ROC for Decision Tree (The Pruned Report Model)
# We need to get probabilities from the model_report
probs_report <- predict(model_report, test_data, type = "prob")[, 2]
roc_tree <- roc(test_data$Target_Next_Direction, probs_report)

# 4. Plot them Comparison
plot(roc_logit, col = "red", main = "The Final Battle: Logistic (Red) vs Tree (Blue)")
plot(roc_tree, col = "blue", add = TRUE)
legend("bottomright", legend = c("Logistic", "Tree"), col = c("red", "blue"), lwd = 2)

# 5. Print the AUC Scores (The Final Scorecard)
print(paste("Logistic AUC:", round(auc(roc_logit), 3)))
print(paste("Tree AUC:    ", round(auc(roc_tree), 3)))



# --- PROJECT B: ALL VISUALS ---
library(rpart)
library(rpart.plot)
library(pROC)
library(ggplot2)

# 1. Load & Prep (Same as before)
forex_data <- read.csv("eurusd.csv", header = TRUE) 
forex_data$Target_Range <- c(forex_data$Range[2:nrow(forex_data)], NA)
forex_data$Direction <- ifelse(forex_data$Close > forex_data$Open, 1, 0)
forex_data$Target_Direction <- c(forex_data$Direction[2:nrow(forex_data)], NA)
forex_data <- na.omit(forex_data)
forex_data <- forex_data[!duplicated(forex_data$Date), ]

# Split
cutoff <- floor(0.7 * nrow(forex_data))
train_data <- forex_data[1:cutoff, ]
test_data  <- forex_data[(cutoff + 1):nrow(forex_data), ]

# --- VISUAL 4: TIME SERIES SNAPSHOT ---
# Just plotting the first 100 hours to show volatility
ggplot(train_data[1:100, ], aes(x = 1:100, y = Close)) +
  geom_line(color = "blue") +
  geom_ribbon(aes(ymin = Low, ymax = High), alpha = 0.2) +
  labs(title = "Figure 4: EUR/USD Hourly Volatility Snapshot (First 100 Hours)",
       x = "Time (Hours)", y = "Price Level") +
  theme_minimal()

# --- TRAIN MODEL (For Visuals 5 & 6) ---
model_tree <- rpart(as.factor(Target_Direction) ~ RSI + MACD + BB_Width + Range, 
                    data = train_data, method = "class",
                    control = rpart.control(cp = 0.001, maxdepth = 3))

# --- VISUAL 5: FEATURE IMPORTANCE ---
# This extracts "Variable Importance" score from the model
importance <- model_tree$variable.importance
importance_df <- data.frame(Feature = names(importance), Score = importance)

ggplot(importance_df, aes(x = reorder(Feature, Score), y = Score)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Figure 5: Feature Importance (What drives the prediction?)",
       x = "Indicator", y = "Importance Score") +
  theme_minimal()

# --- VISUAL 6: THE STRATEGY TREE ---
rpart.plot(model_tree, type = 4, extra = 104, tweak = 1.2, box.palette = "GnBu",
           main = "Figure 6: The 'Forex Sniper' Decision Logic")

# --- VISUAL 7: ROC CURVE ---
# Train Logistic
model_linear <- glm(as.factor(Target_Direction) ~ RSI + MACD + BB_Width + Range, 
                    data = train_data, family = "binomial")
probs_linear <- predict(model_linear, test_data, type = "response")
probs_tree <- predict(model_tree, test_data, type = "prob")[, 2]

roc_logit <- roc(test_data$Target_Direction, probs_linear)
roc_tree <- roc(test_data$Target_Direction, probs_tree)

plot(roc_logit, col = "red", main = "Figure 7: ROC Comparison (Logistic vs Tree)")
plot(roc_tree, col = "blue", add = TRUE)
legend("bottomright", legend = c("Logistic (Linear)", "Tree (Non-Linear)"), 
       col = c("red", "blue"), lwd = 2)



# --- DIAGNOSTIC & FIX ---

# 1. Check if data exists (Look at the Console output!)
print(paste("Rows in Train Data:", nrow(train_data)))
print("First 5 rows of Close Price:")
print(head(train_data$Close))

# 2. The Safer Plotting Code
# We use 'head(train_data, 100)' to safely get the first 100 rows
# We use 'as.numeric' to ensure the price is treated as a number, not text
subset_data <- head(train_data, 100)

ggplot(subset_data, aes(x = 1:nrow(subset_data), y = as.numeric(Close))) +
  geom_line(color = "blue") +
  geom_ribbon(aes(ymin = as.numeric(Low), ymax = as.numeric(High)), alpha = 0.2) +
  labs(title = "Figure 4: EUR/USD Hourly Volatility Snapshot (First 100 Hours)",
       x = "Time (Hours)", y = "Price Level") +
  theme_minimal()




# 1. Load the library first
library(ggplot2)

# 2. Define the data subset (First 100 rows)
# We use 'head' to be safe
subset_data <- head(train_data, 100)

# 3. Plot
ggplot(subset_data, aes(x = 1:nrow(subset_data), y = Close)) +
  geom_line(color = "blue") +
  geom_ribbon(aes(ymin = Low, ymax = High), alpha = 0.2) +
  labs(title = "Figure 4: EUR/USD Hourly Volatility Snapshot (First 100 Hours)",
       x = "Time (Hours)", y = "Price Level") +
  theme_minimal()


# 1. Force a new window to open
dev.new() 

# 2. Create the plot object
p <- ggplot(train_data[1:100, ], aes(x = 1:100, y = Close)) +
  geom_line(color = "blue") +
  geom_ribbon(aes(ymin = Low, ymax = High), alpha = 0.2) +
  labs(title = "Figure 4: EUR/USD Hourly Volatility Snapshot",
       x = "Time (Hours)", y = "Price Level") +
  theme_minimal()

# 3. Explicitly print it (This forces it to appear)
print(p)


# --- THE RESCUE SCRIPT: GENERATE ALL PROJECT B VISUALS ---

# 1. Setup Libraries
if(!require(ggplot2)) install.packages("ggplot2")
if(!require(rpart)) install.packages("rpart")
if(!require(rpart.plot)) install.packages("rpart.plot")
if(!require(pROC)) install.packages("pROC")
library(ggplot2); library(rpart); library(rpart.plot); library(pROC)

# 2. Reload & Clean Data (Starting Fresh)
# REPLACE "clean_forex.csv" with your actual file name if different!
raw_data <- read.csv("clean_forex.csv", header = TRUE) 

# Create Targets (Shifted for prediction)
raw_data$Target_Range <- c(raw_data$Range[2:nrow(raw_data)], NA)
raw_data$Direction <- ifelse(raw_data$Close > raw_data$Open, 1, 0)
raw_data$Target_Direction <- c(raw_data$Direction[2:nrow(raw_data)], NA)

# Remove NAs and Duplicates
clean_data <- na.omit(raw_data)
clean_data <- clean_data[!duplicated(clean_data$Date), ]

# Split Data (70/30)
cutoff <- floor(0.7 * nrow(clean_data))
train_data <- clean_data[1:cutoff, ]
test_data  <- clean_data[(cutoff + 1):nrow(clean_data), ]

# --- FIGURE 4: TIME SERIES SNAPSHOT ---
# We force a new window and check for data first
if(nrow(train_data) > 0) {
  dev.new() # Opens a new window
  p4 <- ggplot(head(train_data, 100), aes(x = 1:100, y = Close)) +
    geom_line(color = "blue") +
    geom_ribbon(aes(ymin = Low, ymax = High), alpha = 0.2) +
    labs(title = "Figure 4: EUR/USD Volatility Snapshot (First 100 Hours)",
         x = "Time (Hours)", y = "Price Level") +
    theme_minimal()
  print(p4)
} else {
  print("ERROR: train_data is empty. Check your CSV filename.")
}

# --- TRAIN MODEL (Required for Figures 5, 6, 7) ---
model_tree <- rpart(as.factor(Target_Direction) ~ RSI + MACD + BB_Width + Range, 
                    data = train_data, method = "class",
                    control = rpart.control(cp = 0.001, maxdepth = 3))

# --- FIGURE 5: FEATURE IMPORTANCE ---
dev.new()
importance <- model_tree$variable.importance
importance_df <- data.frame(Feature = names(importance), Score = importance)
p5 <- ggplot(importance_df, aes(x = reorder(Feature, Score), y = Score)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Figure 5: Feature Importance (What drives the model?)",
       x = "Indicator", y = "Importance Score") +
  theme_minimal()
print(p5)

# --- FIGURE 6: THE STRATEGY TREE ---
dev.new()
rpart.plot(model_tree, type = 4, extra = 104, tweak = 1.2, box.palette = "GnBu",
           main = "Figure 6: The 'Forex Sniper' Logic")

# --- FIGURE 7: ROC CURVE ---
dev.new()
# Train Logistic Baseline
model_logit <- glm(as.factor(Target_Direction) ~ RSI + MACD + BB_Width + Range, 
                   data = train_data, family = "binomial")
probs_logit <- predict(model_logit, test_data, type = "response")
probs_tree  <- predict(model_tree, test_data, type = "prob")[, 2]

roc_logit <- roc(test_data$Target_Direction, probs_logit)
roc_tree  <- roc(test_data$Target_Direction, probs_tree)

plot(roc_logit, col = "red", main = "Figure 7: ROC Comparison")
plot(roc_tree, col = "blue", add = TRUE)
legend("bottomright", legend = c("Logistic (Red)", "Tree (Blue)"), 
       col = c("red", "blue"), lwd = 2)


# --- THE FINAL RESCUE SCRIPT (PROJECT B) ---

# 1. Setup Libraries
if(!require(ggplot2)) install.packages("ggplot2")
if(!require(rpart)) install.packages("rpart")
if(!require(rpart.plot)) install.packages("rpart.plot")
if(!require(pROC)) install.packages("pROC")
library(ggplot2); library(rpart); library(rpart.plot); library(pROC)

# 2. Load Data (Using your code settings)
# Ensure "eurusd.csv" is in your working directory!
forex_data <- read.csv("eurusd.csv", header = TRUE, stringsAsFactors = FALSE)

# 3. Clean & Prepare Targets (Your exact logic)
clean_forex <- forex_data[, c("Date", "Open", "High", "Low", "Close", 
                              "rsi14_H1", "macd_H1", "volatility_bbw_ta_H1")]
colnames(clean_forex) <- c("Date", "Open", "High", "Low", "Close", "RSI", "MACD", "BB_Width")

clean_forex$Range <- clean_forex$High - clean_forex$Low
clean_forex$Direction <- ifelse(clean_forex$Close > clean_forex$Open, 1, 0)

# Shift Targets (The "Time Shift")
clean_forex$Target_Next_Range <- c(clean_forex$Range[2:nrow(clean_forex)], NA)
clean_forex$Target_Next_Direction <- c(clean_forex$Direction[2:nrow(clean_forex)], NA)

# Final Cleaning
clean_forex <- na.omit(clean_forex)
clean_forex <- clean_forex[!duplicated(clean_forex$Date), ]

# 4. Split Data (70/30)
cutoff_index <- floor(0.7 * nrow(clean_forex))
train_data <- clean_forex[1:cutoff_index, ]
test_data  <- clean_forex[(cutoff_index + 1):nrow(clean_forex), ]

# --- FIGURE 4: TIME SERIES SNAPSHOT ---
# We use dev.new() to force a new window
dev.new()
# Plotting just the first 100 hours of the clean training data
ggplot(head(train_data, 100), aes(x = 1:100, y = Close)) +
  geom_line(color = "blue") +
  geom_ribbon(aes(ymin = Low, ymax = High), alpha = 0.2) +
  labs(title = "Figure 4: EUR/USD Volatility Snapshot (First 100 Hours)",
       x = "Time (Hours)", y = "Price Level") +
  theme_minimal()

# --- RE-TRAIN THE MODEL (To ensure Figures 5, 6, 7 work) ---
# We use your 'model_report' settings (maxdepth = 3)
model_report <- rpart(Target_Next_Direction ~ RSI + MACD + BB_Width + Range, 
                      data = train_data, 
                      method = "class",
                      control = rpart.control(cp = 0.001, maxdepth = 3))

# --- FIGURE 5: FEATURE IMPORTANCE ---
dev.new()
importance <- model_report$variable.importance
importance_df <- data.frame(Feature = names(importance), Score = importance)

ggplot(importance_df, aes(x = reorder(Feature, Score), y = Score)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Figure 5: Feature Importance",
       x = "Indicator", y = "Importance Score") +
  theme_minimal()

# --- FIGURE 6: THE STRATEGY TREE ---
dev.new()
rpart.plot(model_report, 
           type = 4, 
           extra = 104, 
           tweak = 1.2, 
           box.palette = "GnBu", 
           main = "Figure 6: The 'Forex Sniper' Logic")

# --- FIGURE 7: ROC CURVE ---
dev.new()
# Train Logistic Baseline (for comparison)
model_logit <- glm(Target_Next_Direction ~ RSI + MACD + BB_Width + Range, 
                   data = train_data, 
                   family = "binomial")

probs_logit <- predict(model_logit, test_data, type = "response")
probs_tree  <- predict(model_report, test_data, type = "prob")[, 2]

roc_logit <- roc(test_data$Target_Next_Direction, probs_logit)
roc_tree  <- roc(test_data$Target_Next_Direction, probs_tree)

plot(roc_logit, col = "red", main = "Figure 7: ROC Comparison")
plot(roc_tree, col = "blue", add = TRUE)
legend("bottomright", legend = c("Logistic (Red)", "Tree (Blue)"), 
       col = c("red", "blue"), lwd = 2)




# --- ESSENTIAL SETUP ---

# Load libraries
library(ggplot2)
library(rpart)
library(pROC)

# Load the raw dataset
forex_data <- read.csv("eurusd.csv", header = TRUE, stringsAsFactors = FALSE)

# Select and Rename columns
clean_forex <- forex_data[, c("Date", "Open", "High", "Low", "Close", 
                              "rsi14_H1", "macd_H1", "volatility_bbw_ta_H1")]
colnames(clean_forex) <- c("Date", "Open", "High", "Low", "Close", "RSI", "MACD", "BB_Width")

# Create Targets and perform the "Time Shift"
clean_forex$Range <- clean_forex$High - clean_forex$Low
clean_forex$Direction <- ifelse(clean_forex$Close > clean_forex$Open, 1, 0)

# Shift vectors to predict the NEXT hour
clean_forex$Target_Next_Range <- c(clean_forex$Range[2:nrow(clean_forex)], NA)
clean_forex$Target_Next_Direction <- c(clean_forex$Direction[2:nrow(clean_forex)], NA)

# Final Cleaning: Remove NAs and Duplicates
clean_forex <- na.omit(clean_forex)
clean_forex <- clean_forex[!duplicated(clean_forex$Date), ]

# Create the 70/30 Train/Test Split
set.seed(123)
cutoff_index <- floor(0.7 * nrow(clean_forex))
train_data <- clean_forex[1:cutoff_index, ]
test_data  <- clean_forex[(cutoff_index + 1):nrow(clean_forex), ]

# Convert the Classification Target to a Factor
train_data$Target_Next_Direction <- as.factor(train_data$Target_Next_Direction)
test_data$Target_Next_Direction  <- as.factor(test_data$Target_Next_Direction)

print("Environment Ready: Data loaded and split successfully.")



# Linear Regression (Baseline)
model_linear <- lm(Target_Next_Range ~ RSI + MACD + BB_Width + Range, data = train_data)
rmse_linear  <- sqrt(mean((test_data$Target_Next_Range - predict(model_linear, test_data))^2))

# Logistic Regression (Baseline)
model_logit  <- glm(Target_Next_Direction ~ RSI + MACD + BB_Width + Range, 
                    data = train_data, family = "binomial")

# Decision Trees
model_tree   <- rpart(Target_Next_Range ~ RSI + MACD + BB_Width + Range, 
                      data = train_data, method = "anova")
rmse_tree    <- sqrt(mean((test_data$Target_Next_Range - predict(model_tree, test_data))^2))

model_roc_tree <- rpart(Target_Next_Direction ~ RSI + MACD + BB_Width + Range, 
                        data = train_data, method = "class")
roc_tree       <- roc(test_data$Target_Next_Direction, 
                      predict(model_roc_tree, test_data, type = "prob")[, 2])



# --- PROJECT B EXPANSION: TESTING MORE ALGORITHMS ---

# 1. Install and Load Random Forest Library
if(!require(randomForest)) install.packages("randomForest")
library(randomForest)

# 2. Random Forest Regression (Predicting Volatility)
# We use 500 trees to try and capture complex patterns
set.seed(123)
model_rf_reg <- randomForest(Target_Next_Range ~ RSI + MACD + BB_Width + Range, 
                             data = train_data, ntree = 500)

# Evaluate RF Regression
preds_rf_reg <- predict(model_rf_reg, test_data)
rmse_rf_reg <- sqrt(mean((test_data$Target_Next_Range - preds_rf_reg)^2))

# 3. Random Forest Classification (Predicting Direction)
# Note: Ensure Target_Next_Direction is a factor
set.seed(123)
model_rf_class <- randomForest(Target_Next_Direction ~ RSI + MACD + BB_Width + Range, 
                               data = train_data, ntree = 500)

# Evaluate RF Classification
probs_rf_class <- predict(model_rf_class, test_data, type = "prob")[, 2]
roc_rf <- roc(test_data$Target_Next_Direction, probs_rf_class)
auc_rf <- auc(roc_rf)

# --- 4. THE COMPARISON TABLE ---
# This directly addresses the lecturer's feedback on "comparison of algorithms"
performance_comparison <- data.frame(
  Task = c("Regression (RMSE)", "Regression (RMSE)", "Regression (RMSE)", 
           "Classification (AUC)", "Classification (AUC)", "Classification (AUC)"),
  Algorithm = c("Linear Model", "Decision Tree", "Random Forest", 
                "Logistic Reg", "Decision Tree", "Random Forest"),
  Score = c(round(rmse_linear, 6), round(rmse_tree, 6), round(rmse_rf_reg, 6),
            round(auc(roc_logit), 3), round(auc(roc_tree), 3), round(auc_rf, 3))
)

print(performance_comparison)

# --- FIX: Generate ROC objects for Comparison ---

# 1. Get probabilities for Logistic Regression
probs_logit <- predict(model_logit, test_data, type = "response")
roc_logit <- roc(test_data$Target_Next_Direction, probs_logit)

# 2. Get probabilities for the Classification Tree
# (Ensure you use the classification version of the tree)
model_class_tree <- rpart(Target_Next_Direction ~ RSI + MACD + BB_Width + Range, 
                          data = train_data, method = "class")
probs_tree <- predict(model_class_tree, test_data, type = "prob")[, 2]
roc_tree <- roc(test_data$Target_Next_Direction, probs_tree)

print("ROC objects created. You can now run the Random Forest code.")



# --- REGENERATE FIGURE 7 WITH ALL 3 MODELS ---

# Plot the first curve (Logistic - Red)
plot(roc_logit, col = "red", main = "Figure 7: ROC Curve Comparison (3 Models)")

# Add the second curve (Decision Tree - Blue)
plot(roc_tree, col = "blue", add = TRUE)

# Add the third curve (Random Forest - Green)
plot(roc_rf, col = "darkgreen", add = TRUE)

# Add the updated legend
legend("bottomright", 
       legend = c("Logistic (Red)", "Tree (Blue)", "Random Forest (Green)"), 
       col = c("red", "blue", "darkgreen"), 
       lwd = 2)



# --- LECTURER FEEDBACK: TECHNICAL INDICATOR CHARTS ---

# 1. Install and load the gridExtra package to stack the plots
if(!require(gridExtra)) install.packages("gridExtra")
library(gridExtra)
library(ggplot2)

# 2. Subset the first 100 hours for a clean, visible snapshot
plot_data <- clean_forex[1:100, ]
plot_data$Time <- 1:nrow(plot_data) # Create a simple x-axis

# 3. Create the Price Plot
p_price <- ggplot(plot_data, aes(x = Time, y = Close)) +
  geom_line(color = "blue", linewidth = 0.8) +
  theme_minimal() +
  labs(title = "EUR/USD Price Action", x = "", y = "Price Level")

# 4. Create the RSI Plot (Adding your Sniper logic lines at 79 and 32)
p_rsi <- ggplot(plot_data, aes(x = Time, y = RSI)) +
  geom_line(color = "darkorange", linewidth = 0.8) +
  geom_hline(yintercept = 79, linetype = "dashed", color = "red") +
  geom_hline(yintercept = 32, linetype = "dashed", color = "darkgreen") +
  theme_minimal() +
  labs(title = "RSI (Momentum & Sniper Thresholds)", x = "", y = "RSI")

# 5. Create the MACD Plot
p_macd <- ggplot(plot_data, aes(x = Time, y = MACD)) +
  geom_line(color = "purple", linewidth = 0.8) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  theme_minimal() +
  labs(title = "MACD (Trend Direction)", x = "", y = "MACD Value")

# 6. Create the Bollinger Band Width Plot
p_bbw <- ggplot(plot_data, aes(x = Time, y = BB_Width)) +
  geom_line(color = "darkred", linewidth = 0.8) +
  theme_minimal() +
  labs(title = "Bollinger Band Width (Volatility)", x = "Time (Hours)", y = "BB Width")

# 7. Stack them all together into one professional graphic
combined_plot <- grid.arrange(p_price, p_rsi, p_macd, p_bbw, ncol = 1)

# Save the plot automatically to your working directory so it's high quality
ggsave("Figure_4_Indicators.png", plot = combined_plot, width = 8, height = 10, dpi = 300)