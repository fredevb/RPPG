import mediapipe as mp
import json
from project_handlers.project_info import ProjectInfo
from project_handlers.project_triangulation_handler import ProjectTriangulationHandler
from lib.preprocessing.triangulation import closest_point_idx
import numpy as np
import cv2
from scipy.spatial import Delaunay
from lib.preprocessing.detection import FaceMeshDetection
import matplotlib.pyplot as plt

info = ProjectInfo()
triangulation_root = info.traingulation_root

triangulation_handler = ProjectTriangulationHandler(triangulation_root)
triangulation_path = triangulation_handler.get_triangulation_path('reduced')

reference_image_path = 'resources/referenceX.jpg'

reference_point_color = [0,255,0]

image = cv2.cvtColor(cv2.imread(reference_image_path), cv2.COLOR_BGR2RGB)

reference_points = []
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        if image[i,j,0] < 10 and image[i,j,1] > 240 and image[i,j,2] < 10:
            reference_points.append(np.array([j, i]))
        #if (image[i,j,0] == reference_point_color[0]
        #    and image[i,j,1] == reference_point_color[1]
        #    and image[i,j,2] == reference_point_color[2]):
        #    reference_points.append(np.array([i,j]))
face_mesh_detection = FaceMeshDetection()
all_points = np.array(face_mesh_detection.detect_face_mesh(image))
point_all_point_idxs = []
for p in reference_points:
    point_all_point_idxs.append(closest_point_idx(all_points, p))
point_all_point_idxs = np.array(point_all_point_idxs)
points = all_points[point_all_point_idxs]

strictly_outside_points = None

points_triangles_idxs = Delaunay(points).simplices.tolist()
triangles = np.array([point_all_point_idxs[t] for t in points_triangles_idxs]).tolist()

with open(triangulation_path, 'w') as f:
    json.dump(triangles, f)

print('Generated Triangulation')

for p in points:
    image = cv2.circle(image, p, 2, (255, 0, 0), 2)

plt.imshow(image)
plt.show()