# click 20 pictures of each person coming in front of the camera and save them as numpy array

import cv2
import numpy as np

# create camera obj
cam = cv2.VideoCapture(0)  # 0 shows the index of the camera to be used.
# read image from camera obj
# Ask the name
fileName = input("Enter the name of the person :")
dataset_path = "./data/"
model = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
offset = 20

faceData = []
skip = 0
while True:  # Loop basically turns everything to a Video
    success, img = cam.read()
    if not success:
        print("Reading from Camera Failed!")
    # Store the gray image to reduce on space
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = model.detectMultiScale(img, 1.3, 5)
    faces = sorted(faces, key=lambda f: f[2] * f[3])  # pick the face with the largest bounding box
    if len(faces) > 0:
        f = faces[-1]
        x, y, w, h = f
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # get the cropped face and store
        cropped_face = img[y - offset:y + h + offset, x - offset:x + w + offset]
        # resize
        cropped_face = cv2.resize(cropped_face, (100, 100))
        skip += 1
        if skip % 10 == 0:
            faceData.append(cropped_face)
            print("Saved so far " + str(len(faceData)))
    cv2.imshow("Image Window", img)
    # cv2.imshow("Cropped Face", cropped_face)
    key = cv2.waitKey(1)  # 1msecs Time of Showing
    if key == ord('q'):
        break

faceData = np.asarray(faceData)
print(faceData.shape)
m = faceData.shape[0]
faceData = faceData.reshape((m, -1))
print(faceData.shape)

# Save on the Disk as np array
filepath = dataset_path + fileName + ".npy"
np.save(filepath, faceData)
print("Data Saved Successfully" + filepath)
# Release Camera and Destroy Window
cam.release()
cv2.destroyAllWindows()
