import cv2 as cv
import numpy as np

def canny(img):
    return cv.Canny(img,50,150)

def mask(img):
    b,l=img.shape
    mask = np.zeros(img.shape[:2], dtype="uint8")
    cv.fillPoly(mask, np.array([[(int(0.25*l),b),(int(l*0.55), int(0.6*b)), (int(l*0.65), int(0.6*b)), (int(0.9*l),b)]]), 255)

    return cv.bitwise_and(img,mask)


def avg_lane_corners(lines,b):
    left_m=[]
    right_m=[]
    left_c=[]
    right_c=[]
    if lines is not None:
        for points in lines:
            x1,y1,x2,y2=points[0]
            m,c=np.polyfit([x1,x2],[y1,y2],1)
            if m>=0:
                right_m.append(m)
                right_c.append(c)
            else:
                left_m.append(m)
                left_c.append(c)
                

    if len(left_m)==0 or len(right_m)==0:
        return None
    
    m_l,c_l=np.average(left_m),np.average(left_c)
    m_r,c_r=np.average(right_m),np.average(right_c)


    def l_line(y):
        return (y-c_l)/m_l
    def r_line(y):
        return (y-c_r)/m_r

    y_10=0.6*b
    corners=np.array([[(int(l_line(b)),b),(int(l_line(y_10)),int(y_10)),(int(r_line(y_10)),int(y_10)),(int(r_line(b)),b)]])
    return corners

def mark_line(img):
    b,l=img.shape[:2]
    # inp=np.float32([[480,1080],[850,780],[1450,780],[1880,1080]])
    # out=np.float32([[0,300],[0,0],[300,0],[300,300]])
    # matrix = cv.getPerspectiveTransform(inp, out)
    # result = cv.warpPerspective(img, matrix, (300, 300))
    grey=cv.cvtColor(img,cv.COLOR_RGB2GRAY)
    grey=cv.GaussianBlur(grey,(5,5),0)
    edge=canny(grey)
    roi=mask(edge)
    lines = cv.HoughLinesP(roi, 1, np.pi / 180, 20,minLineLength=5,maxLineGap=60)

    corners=avg_lane_corners(lines,b)
    lane=np.zeros(img.shape,'uint8')
    cv.fillPoly(lane,corners,(0,255,0))
    cv.polylines(img,corners,True,(0,0,255),2)
    return cv.addWeighted(img,0.8,lane,0.3,1) 
        
def findLane(vid,rec):
    cap=cv.VideoCapture(vid)
    fps=int(round(cap.get(5),0))
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_size = (frame_width,frame_height)
    output = cv.VideoWriter(rec, cv.VideoWriter_fourcc('M','J','P','G'), fps,frame_size)
    while cap.isOpened():
        t,frame=cap.read()
        if t:
            img=mark_line(frame)
            output.write(img)
            cv.imshow('Marked',img)
            if cv.waitKey(fps) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    output.release()
    cv.destroyAllWindows()

findLane('./videoplayback.mp4','Lane.avi')