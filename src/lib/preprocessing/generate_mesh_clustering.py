import cv2
from preprocessing.triangulation import polygon_points_from_indexes
from detection import FaceMeshDetection
from sklearn.cluster import KMeans
import json
import numpy as np
import matplotlib.pyplot as plt

# Load project information.
with open('project_info.json', 'r') as f:
    info = json.load(f)

n_clusters = 25
sample_image_path = 'resources/sample_image.png'
triangulation_path = info['triangulation_path']

with open(triangulation_path) as f:
    triangles = json.load(f)

sample_image = cv2.imread(sample_image_path)

points = FaceMeshDetection().detect_face_mesh(sample_image)

triangle_points = polygon_points_from_indexes(points, triangles)
centers = np.average(triangle_points, axis=2)

kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
kmeans.fit(centers)
pred = kmeans.predict(centers)

clusters_triangle_points = [[] for _ in range(n_clusters)]
for triangle_idx, cluster_idx in enumerate(pred):
    clusters_triangle_points[cluster_idx].append(triangle_points[triangle_idx])

for cluster_idx in range(n_clusters):
    clusters_triangle_points[cluster_idx] = np.array(clusters_triangle_points[cluster_idx])
clusters_triangle_points = np.asarray(clusters_triangle_points)

polygons = []

for cluster_triangle_points in clusters_triangle_points:
    polygon = []
    d = {}
    for tp in cluster_triangle_points:
        for point in tp:
            point = tuple(point.tolist())
            if point not in d:
                d[point] = 0
            d[point] = d[point] + 1
    for point, count in d.items():
        if count == 2:
            polygon.append(np.array(list(point)))
    polygons.append(np.array(polygon))

polygons = np.array(polygons)

plt.imshow(sample_image)
#print(polygons[0].T[0])
#print(polygons.T[0][0])
#plt.plot(polygons[0].T[0], polygons[0].T[1])

e = [[] for _ in range(n_clusters)]
for triangle_idx, cluster_idx in enumerate(pred):
    e[cluster_idx].append(triangles[triangle_idx])

#print(clusters_triangle_points[0][:][:][0])
#print(clusters_triangle_points[0])

for u in e:
    plt.triplot(np.array(triangle_points)[:,:,0].flatten(), np.array(triangle_points)[:,:,1].flatten(), u)
plt.show()

#with open('resources/polygons.json', 'w') as f:
#    json.dump(polygons, f)