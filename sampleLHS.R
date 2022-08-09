library(lhs)
n = 8
d = 7
X = maximinLHS(n, d)
write.csv(X, paste('data/lhs_', toString(n), '.csv', sep=''))