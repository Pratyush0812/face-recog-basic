import os
import cv2
import numpy as np

dataset_path = "./data/"
faceData = []
labels = []
nameMap = {}

classId = 0
for f in os.listdir(dataset_path):
    if f.endswith(".npy"):
        nameMap[classId] = f[:-4]
        # X-Value
        dataItem = np.load(dataset_path + f)
        m = dataItem.shape[0]
        print(dataItem.shape)
        faceData.append(dataItem)
        # Y-Value
        target = classId * np.ones((m,))
        classId += 1
        labels.append(target)

# print(faceData)
# print(labels)
print(nameMap)
XT = np.concatenate(faceData, axis=0)
yT = np.concatenate(labels, axis=0).reshape((-1, 1))
print(XT.shape, yT.shape)


# Algorithm
def dist(p, q):
    return np.sqrt(np.sum((p - q) ** 2))


def knn(X, y, xt, k=5):
    m = X.shape[0]
    dlist = []

    for i in range(m):
        d = dist(X[i], xt)
        dlist.append((d, y[i]))

    dlist = sorted(dlist)
    dlist = np.array(dlist[:k])
    labels = dlist[:, 1]

    labels, cnts = np.unique(labels, return_counts=True)
    idx = cnts.argmax()
    pred = labels[idx]

    return int(pred)

#Prediction

offset = 20
cam = cv2.VideoCapture(0)
model = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

while True:  # Loop basically turns everything to a Video
    success, img = cam.read()
    if not success:
        print("Reading from Camera Failed!")

    faces = model.detectMultiScale(img, 1.3, 5)
    for f in faces:
        x, y, w, h = f
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # get the cropped face and store
        cropped_face = img[y - offset:y + h + offset, x - offset:x + w + offset]
        # resize
        cropped_face = cv2.resize(cropped_face, (100, 100))
        #Predict Name
        classPredicted = knn(XT,yT,cropped_face.flatten())
        namePredicted = nameMap[classPredicted]
        #Display
        cv2.putText(img, namePredicted, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Prediction Window",img)
    key = cv2.waitKey(1)  # 1msecs Time of Showing
    if key == ord('q'):
        break