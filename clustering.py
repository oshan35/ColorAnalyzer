import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import MeanShift, estimate_bandwidth
from utils.mean_shift_cosine_gpu import MeanShiftCosine


def meanshift(image):

    """
    :param image:
    :return: array that contain clustered labels and varies details about the cluster
    """
    bandwidth = estimate_bandwidth(image, quantile=0.2, n_samples=500)

    meanshift = MeanShift(bandwidth=bandwidth, bin_seeding=True, n_jobs=-1)
    # labels = meanshift.fit_predict(image)
    meanshift = MeanShiftCosine(bandwidth=bandwidth, GPU=True)
    labels = meanshift.fit_predict(image)

    return [meanshift, labels, meanshift.cluster_centers_]



def sillhouttle_analysis(image,clusters,labels, n_clusters):

    """

    :param image:
    :param clusters:
    :param labels:
    :param n_clusters:
    :return: Generating the cluster analysis results
    left hand side plot shows the sillhouttle values and right hand side plot shows how clusters
    have been divided in 2d plot

    note : It might take little bit of time to generate the plot because there are lot of calculations
    happening in the process
    """
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(image) + (n_clusters + 1) * 10])
    silhouette_avg = silhouette_score(image, labels)

    sample_silhouette_values = silhouette_samples(image, labels)
    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    colors = cm.nipy_spectral(labels.astype(float) / n_clusters)
    ax2.scatter(
        image[:, 0], image[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
    )

    # Labeling the clusters
    centers = clusters.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="o",
        c="white",
        alpha=1,
        s=200,
        edgecolor="k",
    )

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(
        "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
        % n_clusters,
        fontsize=14,
        fontweight="bold",
    )

    plt.show()