import os
import json

class ProjectTriangulationHandler():

    def __init__(self, triangulations_root):
        self.triangulations_root = triangulations_root

    def get_triangulation_path(self, triangulation_name):
        triangulation_path = os.path.join(self.triangulations_root, triangulation_name + '.json')
        return triangulation_path
    
    def load_triangulation(self, triangulation_name):
        triangulation_path = self.get_triangulation_path(triangulation_name)
        with open(triangulation_path, 'r') as f:
            return json.load(f)