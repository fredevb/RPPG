from .detection import FaceDetection, FaceMeshDetection
from .triangulation import polygon_points_from_indexes
from .image_utils import get_bounding_box, crop_image, iter_blocks, cut_polygon, get_bounding_box_points

import numpy as np

class FaceMeshExtractor:
    def __init__(self, triangles):
        self.face_mesh_detection = FaceMeshDetection()
        self.triangles = triangles

    def get_polygons(self, image):
        points = self.face_mesh_detection.detect_face_mesh(image)
        if points is None:
            return None
        triangle_points = np.array(polygon_points_from_indexes(points, self.triangles))
        return triangle_points
    
    def extract_regions(self, image):
        polygons = self.get_polygons(image)
        if polygons is None:
            return None
        
        return np.array([
            cut_polygon(image, polygon)
            for polygon 
            in polygons
        ])
    
    def extract_regions_average(self, image):
        regions = self.extract_regions(image)
        return np.average(regions, axis=0)

class FaceGridExtractor:
    def __init__(self, size_x, size_y):
        self.face_detection = FaceDetection()
        self.size_x = size_x
        self.size_y = size_y
    
    def get_polygons(self, image):
        face_box = self.face_detection.detect_face(image)
        if face_box is None:
            return None
        xm, ym, w, h = get_bounding_box(face_box)
        return np.array([get_bounding_box_points(x_min + xm, y_min + ym, h, w) 
            for x_min, y_min, h, w 
            in iter_blocks(w, h, self.size_x, self.size_y)])
        
    def extract_regions(self, image):
        polygons = self.get_polygons(image)
        if polygons is None:
            return None
        return np.array([
            crop_image(image, x_min, y_min, h, w)
            for x_min, y_min, h, w 
            in [get_bounding_box(polygon) for polygon in polygons]
        ])
    
    def extract_regions_average(self, image):
        regions = self.extract_regions(image)
        return np.average(regions, axis=0)