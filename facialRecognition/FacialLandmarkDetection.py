import cv2
import mediapipe as mp

vidcappy = cv2.VideoCapture(0)
vidcappy.set(3,600)
vidcappy.set(4,480)
NUM_FACE = 1

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh

faceMesh = mpFaceMesh.FaceMesh(max_num_faces=NUM_FACE)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

while True:
    success, image = vidcappy.read()
    ret, image2 = vidcappy.read()

    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(image, faceLms, mpFaceMesh.FACEMESH_TESSELATION, drawSpec, drawSpec)
            for id, lm in enumerate(faceLms.landmark):
                print(lm)
                ih, iw, ic = image.shape
                x, y = int(lm.x * iw), int(lm.y * ih)
                cv2.putText(image, str(id), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 255), 1)
                print(id, x, y)


        cv2.imshow("Facial Landmark Detection", image)
        cv2.imshow("Norm", image2)

        escape_key = cv2.waitKey(5)

        # asking user of they want a copy of the image
        # asking user of they want a copy of the image
        if escape_key == 27:
            userInput = input("Would you like your picture take? Type yes or no on the console")
            if (userInput == "yes"):
                cv2.imwrite("outputImage.jpg", image)
                print("Image was taken!")
                break
            elif (userInput == "no"):
                break

vidcappy.release()
cv2.destroyAllWindows()






