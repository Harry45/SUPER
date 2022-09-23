library(lhs)
n = 16
d = 2
# X = maximinLHS(n, d)
X = optimumLHS(n, d, maxSweeps = 5, eps = 1E-6)
write.csv(X, paste('data/lhs_', toString(n), '.csv', sep=''))