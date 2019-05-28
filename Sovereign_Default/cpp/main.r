logy_grid <- as.matrix(read.csv('logy_grid.txt', header = FALSE, sep = ' '))
Py <- as.matrix(read.csv('P.txt', header = FALSE, sep = ' '))

library(Rcpp)

sourceCpp('cpp_armadillo.cpp')

# code above need only to be run once per R session
# there is no need to reload files and recompile sources each time
# if you want to run the function several times
# you only need to run the code below

res <- dmain(logy_grid, Py, 951, 10)
write.table(res$V, file = 'V.txt', row.names = FALSE, col.names = FALSE)
print(res$millis)
