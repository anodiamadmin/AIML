library(ggplot2)
houses = read_excel("D:/AIML/02FoundationsOfDataScience/R_Programming/HousingPrices.xlsx")
summary(houses)

# Scatter Plot
ggplot(data, aes(x = Rooms, y = SellingPrice, color = as.factor(Location))) +
  geom_point(size = 3) +
  labs(title = "Number of Rooms Vs House Selling Price",
       x = "Number of Rooms", 
       y = "Selling Price (in $1000s)", 
       color = "Suburb") +
  scale_color_manual(values = c("blue", "red"), 
                     labels = c("Dandenong (East)", "Sunshine (West)")) +
  theme_minimal()

# Fit multiple regression model
model <- lm(Price ~ Rooms + Location, data = houses)


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

# Add the interaction term (Location * Number of Rooms)
houses$X1X2 <- houses$Location * houses$Rooms

# Fit the multiple linear regression model with the interaction term
model_with_interaction <- lm(Price ~ Location + Rooms + X1X2, data = houses)

# Print the summary of the regression model
summary(model_with_interaction)
