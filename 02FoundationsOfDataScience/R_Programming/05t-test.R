# Load necessary libraries
library(ggplot2)

houses = read_excel("D:/AIML/02FoundationsOfDataScience/R_Programming/HousingPrices.xlsx")
summary(houses)

# Fit multiple regression model
model <- lm(SellingPrice ~ Rooms + Location, data = data)

# Summary of the model to get t-tests and p-values for individual coefficients
summary_model <- summary(model)

# Print the summary which includes t-test results and p-values
summary_model
