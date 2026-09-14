library(readxl)
fashion_mnist_test <- read.csv("C:/Users/ymaza/Downloads/fashionmnist/fashion-mnist_test.csv")
# View(fashion_mnist_test)

mnist_index <- sample(1:10000, 2000, replace = FALSE)

fmnist <- fashion_mnist_test[mnist_index,]

fmnist_train_index <- sample(1:2000, 1000, replace = FALSE)

fmnist.train <- fmnist[fmnist_train_index,]
fmnist.test <- fmnist[-fmnist_train_index,]

View(fmnist.train)

write.table(fmnist.train[-1,], "FashionMNIST2_train_1000.csv", row.names = FALSE,
            col.names = FALSE, sep = ",")
write.table(fmnist.test[-1,], "FashionMNIST2_test_1000.csv", row.names = FALSE,
            col.names = FALSE, sep = ",")

fmnist.test2 <- fashion_mnist_test[-mnist_index,]

