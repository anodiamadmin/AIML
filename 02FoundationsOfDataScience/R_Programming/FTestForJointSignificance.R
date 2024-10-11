library(tidyverse)
library(readr)
library(ggplot2)
library(nlme)
library(lme4)
library(car)
library(readxl)
houses = read_excel("D:\\AIML\\02FoundationsOfDataScience\\R_Programming\\HousingPrices.xlsx")
summary(houses)

# Fit multiple regression model
model <- lm(Price ~ Rooms + Location, data = data)

# Summary of the model to get t-tests and p-values for individual coefficients
summary_model <- summary(model)

# Print the summary which includes t-test results and p-values
summary_model

# Perform F-Test using ANOVA
anova_result <- anova(model)

# View ANOVA table and p-value
anova_result

# Predict selling price for a 9-room house in Melbourne's east (Location = 0)
new_house <- data.frame(Rooms = 9, Location = 0)
predicted_price <- predict(model, newdata = new_house)

# Print the predicted price
predicted_price

# Retrieve the coefficients (beta values)
coefficients <- coef(model)

# Print the beta values
coefficients

