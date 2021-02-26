import sys
import numpy as np
import scipy.io.wavfile

MAX_ITERATION = 30


def getClusterIndex(vector):
    minDist = -1
    closestCentromIndex = -1

    for idx, centroid in enumerate(centroids):
        dist = np.linalg.norm(vector - centroid)
        if dist < minDist or minDist == -1:
            minDist = dist
            closestCentromIndex = idx

    return closestCentromIndex


def calcAvgOfCluster(cluster, cent):
    if len(cluster) != 0:
        return np.add.reduce(cluster) / len(cluster)
    else:
        return cent


def isTheSame(oldCentroid, newCentroid):
    return np.array_equal(oldCentroid, newCentroid)


def roundCentroid(centroid):
    for dim in range(len(centroid)):
        centroid[dim] = round(centroid[dim])


f = open("output.txt", "w")
sample, centroids = sys.argv[1], sys.argv[2]
fs, y = scipy.io.wavfile.read(sample)  # reading
x = np.array(y.copy())
try:
    centroids = np.loadtxt(centroids)
except:
    f.write("Please init valid centroids")
    f.close()
    exit(0)
if len(centroids) == 0:
    f.write("Please init valid centroids")
    f.close()
    exit(0)

clusters = []
lastCentroids = centroids.copy()
isConvergence = False
new_values = y.copy()
# run the algorithm
for iteration in range(MAX_ITERATION):
    # init clusters
    clusters.clear()
    for i in centroids:
        clusters.append([])

    # for each vector in y - assign it to the cluster with most close centroid
    for idx, vector in enumerate(y):
        clusterIndex = getClusterIndex(vector)
        clusters[clusterIndex].append(vector)
        new_values[idx] = centroids[clusterIndex]

    # for each centroid - set it to the average of all the vector inside its cluster
    isConvergence = True  # assume convergence, if find 1 change set to false
    for idx in range(len(centroids)):
        if iteration == 0:  # for the first iteration, round each centroid, necessary for the convergence check
            roundCentroid(centroids[idx])
        newCentroid = calcAvgOfCluster(clusters[idx], centroids[idx])
        # round the new centroid dimension
        roundCentroid(newCentroid)
        if isConvergence is True:
            isConvergence = isTheSame(centroids[idx], newCentroid)
        centroids[idx] = newCentroid

    # print the new centroids values to file
    f.write(f"[iter {iteration}]:{','.join([str(i) for i in centroids])}\n")

    if isConvergence:
        break

scipy.io.wavfile.write("compressed.wav", fs, np.array(new_values, dtype=np.int16))  # saving
f.close()
