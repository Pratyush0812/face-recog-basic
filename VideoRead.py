
import cv2
#create camera obj
cam = cv2.VideoCapture(0) #0 shows the index of the camera to be used.
#read image from camera obj

model = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

while True: #Loop basically turns everything to a Video
    success, img = cam.read()
    if not success:
        print("Reading from Camera Failed!")
    faces = model.detectMultiScale(img,1.3,5)
    for f in faces:
        x, y, w, h = f
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


    cv2.imshow("Image Window",img)
    key = cv2.waitKey(1) #1msecs Time of Showing
    if key == ord('q'):
        break

#Release Camera and Destroy Window
cam.release()
cv2.destroyAllWindows()