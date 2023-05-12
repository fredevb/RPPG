import mediapipe as mp
from .image_utils import sub_bounding_box, percentage_coords_to_image_coords, get_bounding_box_points

class FaceDetection:
    def __init__(self, zoom = (1,1)):
        self.face_detection = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
        self.zoom = zoom

    def detect_face(self, image):
        detections = self.face_detection.process(image).detections
        if len(detections) == 1:
            for face_no, face in enumerate(detections):
                face_data = face.location_data
                bounding_box = face_data.relative_bounding_box
                (xm, ym, w, h) = sub_bounding_box(image.shape[1], image.shape[0], bounding_box.xmin, bounding_box.ymin,
                                                 bounding_box.width, bounding_box.height, zoom=self.zoom)
                return get_bounding_box_points(xm, ym, w, h)
        return None
    
class FaceMeshDetection:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5, refine_landmarks=True)

    def detect_face_mesh(self, image):
        results = self.face_mesh.process(image)
        landmark_list = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, landmark in enumerate(face_landmarks.landmark):
                    landmark_list.append([landmark.x, landmark.y])
        else:
            return None
        return percentage_coords_to_image_coords(landmark_list, image.shape[1], image.shape[0])