import cv2 as cv

def mark_cars(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cars = car_classifier.detectMultiScale(gray, 1.1, 3)
    for (x,y,w,h) in cars:
        cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)

    return img

def process(vid,out):
    cap = cv.VideoCapture(vid)
    fps=int(round(cap.get(5),0))
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_size = (frame_width,frame_height)
    output = cv.VideoWriter(out, cv.VideoWriter_fourcc('M','J','P','G'), fps,frame_size)
    while cap.isOpened():
        t,frame=cap.read()
        if t:
            img=mark_cars(frame)
            output.write(img)
            cv.imshow('Marked',img)
            if cv.waitKey(fps) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    output.release()
    cv.destroyAllWindows()

car_classifier = cv.CascadeClassifier('haarcascade_car.xml')
process('cars.mp4','Objects.avi')