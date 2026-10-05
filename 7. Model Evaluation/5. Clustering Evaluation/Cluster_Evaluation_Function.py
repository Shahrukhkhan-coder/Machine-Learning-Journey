from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score)


def evaluate_clustering(X, labels, y_true=None):
    
    results = {}

    # INTERNAL METRICS

    results["Silhouette"] = (silhouette_score(X,labels))

    results["Davies_Bouldin"] = (davies_bouldin_score(X,labels))

    results["Calinski_Harabasz"] = (calinski_harabasz_score(X,labels))

    # EXTERNAL METRICS

    if y_true is not None:

        results["ARI"] = (adjusted_rand_score(y_true,labels))

        results["NMI"] = (normalized_mutual_info_score(y_true,labels))

        results["Homogeneity"] = (homogeneity_score(y_true,labels))

        results["Completeness"] = (completeness_score(y_true,labels))

        results["V_measure"] = (
            v_measure_score(y_true,labels))
        
    return results