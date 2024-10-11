library(ggplot2)
houses = read_excel("D:/AIML/02FoundationsOfDataScience/R_Programming/HousingPrices.xlsx")
summary(houses)

# Fit multiple regression model
model <- lm(Price ~ Rooms + Location, data = data)

# Summary of the model
summary(model)

# Predict selling price for a 9-room house in Melbourne's east (Location = 0)
new_house <- data.frame(Rooms = 9, Location = 0)
predicted_price <- predict(model, newdata = new_house)

# Print the predicted price
predicted_price

# Retrieve the coefficients (beta values)
coefficients <- coef(model)

# Print the beta values
coefficients
