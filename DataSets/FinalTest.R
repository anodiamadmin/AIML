# Load Data
setwd("D:\\AIML\\DataSets")
getwd()
list.files()
colony_counts  <- read.csv("colony_counts.csv")
colony_inputs  <- read.csv("colony_inputs.csv")
colony_outputs <- read.csv("colony_outputs.csv")
# EDA
# head(colony_counts)

# Create Model D1
model_D1 <- lm(weight ~ total.fem + other + total, data = colony_counts)

# Q1 Residual standard error
summary(model_D1)

# Q2 coef total.fem
coef(model_D1)

# Q3 high leverage
sum(hatvalues(model_D1) >= 3*length(coef(model_D1))/nrow(colony_counts))

# Q5 TSS
round(sum((colony_counts$weight - mean(colony_counts$weight))^2), 1)

# Q6 sum of the squared standardized residuals
round(sum(rstandard(model_D1)^2), 2)

# Q7 log likelihood
round(logLik(model_D1)[1], 2)

# Q8 AIC second part = 2k
2*length(coef(model_D1))

# Q9 Colinearity
install.packages("car")
library(car)
vif(model_D1)
# strong linear relationship (correlation)
cor(colony_counts[, c("total.fem", "other", "total")])

# Create Model D2
model_D2 <- lm(weight ~ total.fem + other, data = colony_counts)
summary(model_D2)

# Q11 D2 lower BIC
BIC(model_D1)
BIC(model_D2)

#---------------------------------

# Q1 ridge Regrsn
library(glmnet)
x <- as.matrix(colony_counts[, c("total.fem", "other", "total")])
y <- colony_counts$weight
lambda_seq <- seq(0.001, 50, by = 0.001)
# Create Model E1
model_E1 <- glmnet(x, y, alpha = 0, lambda = lambda_seq, standardize = TRUE)
try(dev.off(), silent = TRUE)
par(mar = c(4,4,2,1))
beta <- as.matrix(model_E1$beta)
log_lambda <- log(model_E1$lambda)
matplot(log_lambda, t(beta), type = "l", lty = 1, lwd = 2,
        col = c("lightblue", "orange", "darkgreen"),
        xlab = "log(lambda)",
        ylab = "Standardized Coefficients",
        main = "Ridge Regression (Standardized) Coefficients vs log(lambda)",
        ylim = c(-0.001, 0.004),
        cex.main = 0.9, font.main = 1)
grid(col = "lightgray", lty = "dashed")
legend("topright",
       legend = rownames(beta),
       col = c("lightblue", "orange", "darkgreen"),
       lty = 1, lwd = 2, bty = "o")

# Q3 lambda 1-std-err
round(cv.glmnet(x, y, alpha = 0, lambda = lambda_seq)$lambda.1se, 1)

# Q4 train MSE lambda=3
round(mean((y - predict(model_E1, newx = x, s = 3))^2), 3)

# Q5 leave-one-out CV lambda=3 -> estimate test MSE
round(mean(sapply(1:nrow(x), function(i) {
  fit <- glmnet(x[-i, ], y[-i], alpha = 0, lambda = 3)
  (y[i] - predict(fit, newx = x[i, , drop = FALSE], s = 3))^2
})), 2)

# Q5 leave-one-out CV lambda=3 -> max MSEi
round(max(sapply(1:nrow(x), function(i) {
fit <- glmnet(x[-i, ], y[-i], alpha = 0, lambda = 3)
(y[i] - predict(fit, newx = x[i, , drop = FALSE], s = 3))^2
})), 2)

#---------------------------------

# Create Model F1
model_F1 <- glm(cbind(num.fem.working, total.fem - num.fem.working) ~ weight,
                family = binomial,
                data = colony_outputs)
# Create Model F2
model_F2 <- glm(cbind(num.fem.working, total.fem - num.fem.working) ~ weight + behaviour.cost,
                family = binomial,
                data = colony_outputs)

# Q1 Wt coef
round(coef(model_F1)["weight"], 4)

# Q3 deviance
round(deviance(model_F1), 3)

# Q4 critical value of H0
round(qchisq(0.95, df = 1), 2)

# Q5 which model?
D <- deviance(model_F1) - deviance(model_F2)
D

# Q6 F2 numerical
round(50 * predict(model_F2,
                   newdata = data.frame(weight = 2.3525, behaviour.cost = 0.0018474),
                   type = "response"), 2)

#---------------------------------

library(splines)
library(mgcv)
# Create Model G1
model_G1 <- lm(log.mg ~ ns(prey.sec), data = colony_inputs)
# Create Model G2
model_G2 <- gam(log.mg ~ s(insect.count) + s(prey.sec), data = colony_inputs)
# Create Model G3
model_G3 <- gam(log.mg ~ s(insect.count), data = colony_inputs)

# Q1 G1 Fit natural spline 4 knots, location of first (smallest) knot
round(quantile(colony_inputs$prey.sec, probs = 1/5), 5)

# Q2 G1 scatter plot + natural spline
library(splines)
knots_vec <- quantile(colony_inputs$prey.sec, probs = c(1/5, 2/5, 3/5, 4/5))
model_G1 <- lm(log.mg ~ ns(prey.sec, knots = knots_vec), data = colony_inputs)
plot(colony_inputs$prey.sec, colony_inputs$log.mg,
     main = "Model G1: Natural Spline",
     xlab = "prey.sec", ylab = "log.mg",
     pch = 16, col = "black",
     cex.main = 0.9, font.main = 1)
abline(v = seq(floor(min(colony_inputs$prey.sec)/0.00025)*0.00025,
               ceiling(max(colony_inputs$prey.sec)/0.00025)*0.00025,
               by = 0.00025),
       col = "lightgrey", lty = "dashed")
abline(h = seq(floor(min(colony_inputs$log.mg)/0.25)*0.25,
               ceiling(max(colony_inputs$log.mg)/0.25)*0.25,
               by = 0.25),
       col = "lightgrey", lty = "dashed")
xg <- seq(min(colony_inputs$prey.sec), max(colony_inputs$prey.sec), length.out = 500)
yg <- predict(model_G1, newdata = data.frame(prey.sec = xg))
lines(xg, yg, col = "blue", lwd = 2)
legend("topleft",
       legend = c("Data points", "Natural Spline (G1)"),
       col = c("black", "blue"),
       pch = c(16, NA),
       lty = c(NA, 1),
       lwd = c(NA, 2),
       bty = "o")

