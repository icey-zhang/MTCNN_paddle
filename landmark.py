import cv2
import numpy as np
image = cv2.imread('/home/aistudio/widerface/val/48/landmark/0.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
landmark = np.array([0.33,0.68,0.56,0.33,0.67,0.19,0.19,0.37,0.63,0.63])

landmark = landmark.reshape(2, 5).T
print(landmark)
for j in range(5):
    cv2.circle(image, (int(landmark[j, 0]), int(landmark[j, 1])), 2, (0, 255, 255), 1)

image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
cv2.imwrite("/home/aistudio/1.png", image)