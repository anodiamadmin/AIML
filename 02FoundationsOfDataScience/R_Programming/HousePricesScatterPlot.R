library(ggplot2)
setwd = ("D:/AIML/02FoundationsOfDataScience/R_Programming")
houses = read_excel("./HousingPrices.xlsx")
summary(houses)

ggplot(data, aes(x = Rooms, y = SellingPrice, color = as.factor(Location))) +
  geom_point(size = 3) +
  labs(title = "Number of Rooms Vs House Selling Price",
       x = "Number of Rooms", 
       y = "Selling Price (in $1000s)", 
       color = "Suburb") +
  scale_color_manual(values = c("blue", "red"), 
                     labels = c("Dandenong (East)", "Sunshine (West)")) +
  theme_minimal()
