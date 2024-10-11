P_D <- .01
P_DN <- 1-P_D
P_D_Given_Pos <- 0.9*P_D
P_DN_Given_Pos <- 0.9*P_DN

# X~(mu, sigma)
mu <- -2
sigma <- sqrt(9)
pnorm_1 <- pnorm(-1, mu, sigma)
pnorm_2 <- pnorm(-1)
pnorm_3 <- 1 - pnorm(-3.5, mu, sigma)
pnorm_4 <- pnorm(-3.5, mu, sigma, lower.tail = FALSE)
pnorn_5 <- pnorm(7, mu, sigma) - pnorm(6, mu, sigma)

mu <- 1
sigma <- 2
sample_size <- 25
pnorm_6 <- pnorm(30, sample_size*mu, sqrt(sample_size*sigma*sigma))
pnorm_7 <- pnorm(.5)