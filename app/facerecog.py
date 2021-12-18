from imutils import face_utils
from pprint import pprint
import dlib
import numpy as np


class FaceRecog:
    def __init__(self, path_detector, path_landmarks, path_recognizer):
        self.path_detector = path_detector
        self.path_landmarks = path_landmarks
        self.path_recognizer = path_recognizer

    def initialize(self):
        self.cnn_face_detector = dlib.cnn_face_detection_model_v1(
            self.path_detector)
        self.sp = dlib.shape_predictor(self.path_landmarks)
        self.facerec = dlib.face_recognition_model_v1(self.path_recognizer)

        print("using cuda: {}".format(dlib.DLIB_USE_CUDA))

    def getSignatures(self, path_img):

        img = dlib.load_rgb_image(path_img)

        return self.processImage(img)

    def processImage(self, npArr, upscale=1):

        dets = self.cnn_face_detector(npArr, 2)

        if len(dets) <= 0:
            return None, None, None

        # Find the 5 face landmarks we need to do the alignment.
        faces = dlib.full_object_detections()
        landmarks = []
        for i in range(0, len(dets)):
            d = dets[i]
            rect = dlib.rectangle(
                d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom())
            shap = self.sp(npArr, rect)
            landmarks.append(face_utils.shape_to_np(shap))
            faces.append(shap)

        face_chips = dlib.get_face_chips(npArr, faces, size=150)

        face_descriptors = []
        for i in range(0, len(face_chips)):
            image = face_chips[i]
            face_descriptor_from_prealigned_image = np.array(
                self.facerec.compute_face_descriptor(image))
            face_descriptors.append(face_descriptor_from_prealigned_image)

        return face_chips, landmarks, face_descriptors
