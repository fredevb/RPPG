import mediapipe as mp
import json
from project_handlers.project_info import ProjectInfo
from project_handlers.project_triangulation_handler import ProjectTriangulationHandler
from lib.preprocessing.triangulation import triangles_from_edges

info = ProjectInfo()
triangulation_root = info.traingulation_root

triangulation_handler = ProjectTriangulationHandler(triangulation_root)
triangulation_path = triangulation_handler.get_triangulation_path('triangulation')

mp_face_mesh = mp.solutions.face_mesh


edges = list(mp_face_mesh.FACEMESH_TESSELATION)
n = 478

triangles = triangles_from_edges(edges, n)

with open(triangulation_path, 'w') as f:
    json.dump(triangles, f)

print('Generated Triangulation')