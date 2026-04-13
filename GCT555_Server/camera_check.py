import cv2
for i in range(4):
    cap = cv2.VideoCapture(i)
    ret, frame = cap.read()
    if ret and frame is not None:
        cv2.imshow(f'Camera {i}', frame)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()
    else:
        print(f'Camera {i} no frame')
    cap.release()