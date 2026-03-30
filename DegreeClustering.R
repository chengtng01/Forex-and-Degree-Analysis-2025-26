# 1. Load the data
# We use stringsAsFactors = FALSE so R doesn't auto-convert text
dataset <- read.csv("GraduateEmploymentSurveyNTUNUSSITSMUSUSSSUTD.csv", stringsAsFactors = FALSE)

# 2. Check the structure
# This tells us if columns are Numbers (num/int) or Text (chr)
str(dataset)

# 3. Peek at the top rows
head(dataset)


# 1. Filter for the latest year (2022) to ensure fair economic comparison
# We use 'subset' to grab only rows where the year is 2022
clean_data <- subset(dataset, year == 2022)

# 2. Identify the columns we need for clustering
# These are the "Economic Factors" we discussed
cols_to_clean <- c("employment_rate_overall", 
                   "gross_monthly_median", 
                   "basic_monthly_median", 
                   "gross_mthly_25_percentile", 
                   "gross_mthly_75_percentile")

# 3. Force-convert these columns to Numeric
# We use a loop (sapply) to apply 'as.numeric' to all chosen columns at once
clean_data[cols_to_clean] <- sapply(clean_data[cols_to_clean], as.numeric)

# 4. Remove rows with missing data (NA)
# If a degree has no salary data (NA), we cannot cluster it, so we drop it.
clean_data <- na.omit(clean_data)

# 5. Verify the clean
str(clean_data)


# 1. Feature Engineering: Create the specific economic metrics
# Risk = 100% minus the Employment Rate (Higher number = Higher Risk)
clean_data$unemployment_risk <- 100 - clean_data$employment_rate_overall

# Potential = 75th Percentile minus 25th Percentile
# This measures the "Upside" of the degree.
clean_data$salary_potential <- clean_data$gross_mthly_75_percentile - clean_data$gross_mthly_25_percentile

# 2. Select ONLY the columns we want to feed into the algorithm
# We are choosing: Gross Median Salary, Unemployment Risk, and Salary Potential
features <- clean_data[, c("gross_monthly_median", "unemployment_risk", "salary_potential")]

# 3. Standardization (Z-Score Normalization)
# This forces all variables to have Mean = 0 and Variance = 1
# This ensures "Salary" doesn't dominate "Risk" in the calculation.
scaled_matrix <- scale(features)

# 4. Check the result
# You should see a matrix where numbers look like "0.5", "-1.2", etc.
head(scaled_matrix)


# 1. Initialize a generic vector to store the "Error" (WCSS)
wcss <- vector()

# 2. Run a loop: Try K=1, then K=2, up to K=10
for (i in 1:10) {
  # run k-means algorithm
  # centers = i (number of clusters)
  # nstart = 10 (run it 10 times to ensure it's stable)
  kmeans_model <- kmeans(scaled_matrix, centers = i, nstart = 10)
  
  # Store the total error (Total Within-Cluster Sum of Squares)
  wcss[i] <- kmeans_model$tot.withinss
}

# 3. Plot the Elbow Graph
plot(1:10, wcss, type = "b", 
     main = "The Elbow Method",
     xlab = "Number of Clusters (K)",
     ylab = "Total Error (WCSS)",
     col = "blue", pch = 19)


# 1. Set a "seed" so your results are reproducible (if you run it again, you get the same clusters)
set.seed(123)

# 2. Run K-Means with K=3
final_model <- kmeans(scaled_matrix, centers = 3, nstart = 25)

# 3. "Stitch" the results back to the original clean data
# We add a new column called 'cluster' to our original table so we can see which degree belongs where.
clean_data$cluster <- as.factor(final_model$cluster)

# 4. View the results (The Mean Stats for each Cluster)
# This tells us exactly what "Cluster 1", "Cluster 2", and "Cluster 3" represent.
aggregate(clean_data[, c("gross_monthly_median", "unemployment_risk", "salary_potential")], 
          by = list(clean_data$cluster), 
          FUN = mean)


# 1. View the "Elite" Degrees (Cluster 3)
print("--- TIER 1: HIGH PAY / LOW RISK ---")
head(subset(clean_data, cluster == 3)[, c("university", "degree", "gross_monthly_median")], 10)

# 2. View the "Risky" Degrees (Cluster 1)
# This is the most controversial/interesting list
print("--- TIER 3: HIGH RISK (The Market Trap) ---")
head(subset(clean_data, cluster == 1)[, c("university", "degree", "unemployment_risk")], 10)


# Load the library for nice plotting
library(ggplot2)

# Create the plot
ggplot(clean_data, aes(x = unemployment_risk, y = gross_monthly_median, color = cluster)) +
  geom_point(alpha = 0.7, size = 3) +
  labs(title = "The Economic Landscape of Singapore Degrees",
       subtitle = "K-Means Clustering (K=3) of Salary vs. Risk",
       x = "Unemployment Risk (%)",
       y = "Median Gross Monthly Salary (SGD)") +
  theme_minimal() +
  scale_color_manual(values = c("red", "blue", "green")) # Red for Risk, Blue for Safe, Green for Elite



# 1. Load the required library
# If this fails, run: install.packages("cluster") in your console first
library(cluster)

# 2. Calculate the Silhouette Information
# We compare the cluster labels against the mathematical distance between points
sil_info <- silhouette(final_model$cluster, dist(scaled_matrix))

# 3. Print the Average Silhouette Width
# This single number is your "Validation Metric" for the report
avg_width <- mean(sil_info[, 3])
print(paste("Average Silhouette Score:", round(avg_width, 3)))

# 4. Generate the Visual Plot
# This shows how "healthy" each individual cluster is
plot(sil_info, 
     main = "Silhouette Plot for K=3 (Validation)", 
     col = c("red", "blue", "green"), 
     border = NA)


# Load necessary library
library(ggplot2)
library(gridExtra) # This helps arrange two plots side-by-side

# 1. Create the Salary Box Plot
# We want to see the median and the spread of salaries for each cluster
p1 <- ggplot(clean_data, aes(x = cluster, y = gross_monthly_median, fill = cluster)) +
  geom_boxplot(alpha = 0.7) +
  labs(title = "Comparison 1: Salary", 
       subtitle = "Cluster 1 & 2 pay similarly",
       y = "Gross Monthly Salary ($)") +
  theme_minimal() +
  scale_fill_manual(values = c("red", "blue", "green")) +
  theme(legend.position = "none")

# 2. Create the Risk Box Plot
# This is the most important chart in your report
p2 <- ggplot(clean_data, aes(x = cluster, y = unemployment_risk, fill = cluster)) +
  geom_boxplot(alpha = 0.7) +
  labs(title = "Comparison 2: Risk", 
       subtitle = "But Cluster 1 is much riskier",
       y = "Unemployment Risk (%)") +
  theme_minimal() +
  scale_fill_manual(values = c("red", "blue", "green")) +
  theme(legend.position = "none")

# 3. Display them side-by-side
grid.arrange(p1, p2, ncol = 2)


install.packages("gridExtra")