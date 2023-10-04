from .fpn import solver_fpn
import numpy as np
import igraph as ig

def solver_bbd(S, Lambda):
    S_lambda = np.maximum(0, S - Lambda)  # Thresholded matrix

    # Conduct bridge-block decomposition
    S_res = np.copy(S_lambda)
    S_res[S_lambda < 0] = 0
    np.fill_diagonal(S_res, 0)
    S_supp = S_res > 0
    G_thresholded = ig.Graph.Adjacency(S_supp.tolist(), mode="undirected")

    # Compute set of bridges in the thresholded graph
    bridges = G_thresholded.bridges()

    # Remove bridges from the graph
    G_reduced = G_thresholded.copy()
    for bridge in reversed(bridges):
        G_reduced.delete_edges(bridge)

    # Get the indexes for sub-graphs containing more than one node
    Final_components = G_reduced.connected_components().membership
    Final_components_csize = np.bincount(Final_components)

    Subgraphs_index = np.where(Final_components_csize > 1)[0]
    Subgraph_list = [np.where(Final_components == k_index)[0] for k_index in Subgraphs_index]

    # Solve sub-problems individually using FPN

    out_FPN_sub_opt = []
    for i in range(len(Subgraphs_index)):
        sub_index = Subgraph_list[i]
        S_sub = S[sub_index][:, sub_index]
        Lambda_sub = Lambda[sub_index][:, sub_index]
        out_FPN_sub_opt.append(solver_fpn(S_sub, Lambda_sub))

    # Ontain optimal solution

    p = S.shape[0]
    Theta_hat = np.zeros((p, p))

    Edge_array = np.array(G_thresholded.get_edgelist())

    for e in range(Edge_array.shape[0]):
        i = Edge_array[e, 0]
        j = Edge_array[e, 1]
        if Final_components[i] != Final_components[j]:
            Theta_hat[i, j] = -(S_lambda[i, j]) / (S[i, i] * S[j, j] - (S_lambda[i, j] ** 2))
            Theta_hat[j, i] = Theta_hat[i, j]

    for i in range(p):
        if Final_components_csize[Final_components[i]] == 1:
            Theta_ii = 1
            for j in G_thresholded.neighbors(
                    i):  # Assuming neighbors is a function returning neighbors of node i in graph G_res
                Theta_ii += (S_lambda[i, j] ** 2) / (S[i, i] * S[j, j] - S_lambda[i, j] ** 2)
            Theta_ii /= S[i, i]
            Theta_hat[i, i] = Theta_ii
        else:
            Theta_hat[i, i] = 1 / S[i, i]

    k_indices = np.where(Final_components_csize > 1)[0]
    for k in k_indices:
        k_id = np.where(Subgraphs_index == k)[0][0]
        sub_i = Subgraph_list[k_id]
        Theta_sub = out_FPN_sub_opt[k_id]["X_est"]
        Theta_hat[np.ix_(sub_i, sub_i)] = Theta_sub

        for i in sub_i:
            for j in G_thresholded.neighbors(i):
                if Final_components[i] != Final_components[j]:
                    Theta_hat[i, i] += (1 / S[i, i]) * (S_lambda[i, j] ** 2) / (S[i, i] * S[j, j] - S_lambda[i, j] ** 2)

    return Theta_hat
