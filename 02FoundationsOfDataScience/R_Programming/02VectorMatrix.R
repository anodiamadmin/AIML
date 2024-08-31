# Vectors
x1 <- c(6, 2, 5, 3, 9, 4)
x1
gender <- c("male", 'female')
gender
2:7
seq(from=2, to=12, by=3)
seq(from=3, to=5, by=1/3)
seq(from=4, to=7, by=.25)
rep(3, times=7)
rep('USA', times=6)
rep(2:4, times=4)
rep(seq(from=2, to=8, by=3), times=3)
rep(c('Male', "Female"), times=4)
x <- 1:4
y <- c(1, 3, 5, 7)
x+10
y/2
x+y
y^x
y[3]
x[-3]
y[c(2, 4)]
y[-c(2, 4, 3)]
y[y>2]
y[y>=5]

# Matrix
mat <- matrix(c(1, 2, 3, 4, 5, 6, 7, 8, 9), nrow=3, byrow=TRUE)
matrix(c(1, 2, 3, 4, 5, 6, 7, 8, 9), nrow=3, byrow=FALSE)
matrix(c(1, 2, 3, 4, 5, 6, 7, 8, 9), nrow=3)
mat
mat[2, 3]
mat[c(1, 3), 2]
mat[2,]
mat[,1]
mat*10
