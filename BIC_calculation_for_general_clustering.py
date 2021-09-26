def compute_bic(X, clusters):
    """
    Computes the BIC metric for a given clusters

    Parameters:
    -----------------------------------------
    X     :  [N, d] multidimension np array of data points
    clusters : [N] cluster id of each sample
    
    Returns:
    -----------------------------------------
    BIC value
    """
    #number of clusters
    m = len(np.unique(clusters))
    
    # assign centers and labels
    centers = [[np.average(X[clusters == _], axis=0) for _ in range(m)]]
    labels = clusters
    
    # size of the clusters
    n = np.bincount(labels)
    
    #size of data set
    N, d = X.shape

    #compute variance for all clusters beforehand
    cl_var = (1.0 / (N - m) / d) * sum([sum(distance.cdist(X[np.where(labels == i)], [centers[0][i]], 
             'euclidean')**2) for i in range(m)])

    const_term = 0.5 * m * np.log(N) * (d+1)

    BIC = np.sum([n[i] * np.log(n[i]) -
               n[i] * np.log(N) -
             ((n[i] * d) / 2) * np.log(2*np.pi*cl_var) -
             ((n[i] - 1) * d/ 2) for i in range(m)]) - const_term

    return(BIC)
