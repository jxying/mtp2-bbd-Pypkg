import numpy as np
import igraph as ig
from scipy.stats import multivariate_normal
from mtp2bbd import solver_fpn, objective_function, solver_bbd

# Set problem dimension
p = 500

# Build a BA graph
BA_graph = ig.Graph.Barabasi(n=p, m=1, directed=False)

adjacency_matrix = BA_graph.get_adjacency()
adjacency_matrix = np.array(adjacency_matrix.data)

max_eig = np.max(np.linalg.eigvals(adjacency_matrix)).real
A = 1.05 * max_eig * np.eye(p) - adjacency_matrix
inv_A = np.linalg.inv(A)
D = np.diag(np.sqrt(np.diag(inv_A)))
Mtrue = D @ A @ D
Ratio = 5
X = multivariate_normal.rvs(np.zeros(p), np.linalg.inv(Mtrue), size=Ratio * p)

# Compute the sample covariance matrix
S = np.cov(X, rowvar=False)

# Compute the regularization matrix
Theta0_mat = np.zeros((p, p))
for i in range(p):
    for j in range(p):
        if S[i, j] > 0 and i != j:
            Theta0_mat[i, j] = -S[i, j] / (S[i, i] * S[j, j] - S[i, j] * S[i, j])

chi = 0.02
Lambda = chi / (np.abs(Theta0_mat) + 0.0001)
np.fill_diagonal(Lambda, 0)

Theta_bbd = solver_bbd(S, Lambda)
obj_FPN_bbd = objective_function(Theta_bbd, S - Lambda)["value"]
print(obj_FPN_bbd)

# Use FPN without birgde-block decomposition

opts_FPN = {'max_iter': 1e4, 'tol': 1e-10}
out_FPN = solver_fpn(S, Lambda, opts_FPN)
print(objective_function(out_FPN["X_est"], S - Lambda)["value"])