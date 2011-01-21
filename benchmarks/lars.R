library(lars)
#x1 <- as.matrix(read.table("X.csv"))
#y1 <- read.table("y.csv")
x1=matrix(rnorm(1000*200),1000,200)
y1=as.matrix(rnorm(1000))

print("Shape of X: ")
print(dim(x1))
print("Shape of y: ")
print(dim(y1))

write.table(x1, file="X_lars.csv")
write.table(y1, file="y_lars.csv")
system.time(lars(x1, y1))
