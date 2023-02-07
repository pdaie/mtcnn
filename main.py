from PIL import Image
from mtcnn import MTCNN
import cv2

mtcnn = MTCNN()

image = cv2.imread('tom_cruise.jpg')
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

face_boxes, face_probs = mtcnn.detect(rgb_image)
print(face_boxes, face_probs)

face_box = [round(i) for i in face_boxes[0]]
face_image = image[face_box[1]: face_box[3], face_box[0]: face_box[2]]

cv2.imwrite('tom_cruise_face.jpg', face_image)