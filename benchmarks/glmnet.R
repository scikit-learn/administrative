
library(glmnet)
#x1 <- as.matrix(read.table("X.csv"))
#y1 <- read.table("y.csv")
x1=matrix(rnorm(1000*500),1000,500)
y1=as.matrix(rnorm(1000))

print("Shape of X: ")
print(dim(x1))
print("Shape of y: ")
print(dim(y1))

write.table(x1, file="X.csv")
write.table(y1, file="y.csv")
system.time(glmnet(x1, y1))

