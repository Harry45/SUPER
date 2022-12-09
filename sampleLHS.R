library(lhs)
n = 5000
d = 6
X = maximinLHS(n, d)
write.csv(X, paste('data/lhs_', toString(d), 'd_', toString(n), '.csv', sep=''))