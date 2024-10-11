library(ggplot2)
setwd("D:/AIML/02FoundationsOfDataScience/R_Programming/data")
houseData = read_excel("HousePrice.xlsx", sheet = "Datasheet")
summary(houseData)

#ggplot(houseData, aes(x=exp(Rooms), y=Price, color=as.factor(Location))) +
#ggplot(houseData, aes(x=log(Rooms), y=Price, color=as.factor(Location))) +
#ggplot(houseData, aes(x=Rooms, y=Price, color=as.factor(Location))) +
#  geom_point(size = 3) +
#  labs(title = "Number of Rooms Vs House Selling Price",
#       x = "Number of Rooms", 
#       y = "Selling Price (in $1000s)", 
#       color = "Suburb") +
#  scale_color_manual(values = c("blue", "red"), 
#                     labels = c("Dandenong (East)", "Sunshine (West)")) +
#  theme_minimal() + geom_smooth(method = 'lm')
ggplot(houseData, aes(x=Rooms, y=Price)) + geom_point() + geom_smooth(method = 'lm')
reg = lm(Price ~ Rooms + Location, data=houseData)
summary(reg)
