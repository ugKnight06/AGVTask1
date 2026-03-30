import numpy as np
import cv2 as cv 
from scipy.signal import convolve2d

sobel_x=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
sobel_y=np.array([[-1,-2,-1],[0,0,0],[1,2,1]])

def LKanade(old_gray, frame_gray, p0)
  
Ix = convolve2d(old_gray, sobel_x)
Iy = convolve2d(old_gray, sobel_y)
It = frame_gray-old_gray

nextPts = []
for [x, y] in p0: 
    ATA = np.array([[np.sum(Ix@Ix), np.sum(Iy@Iy)],[np.sum(Ix@Iy),np.sum(Iy@Iy)]])
    ATb = np.array([[-np.sum(Ix*It)], 
                    [-np.sum(Iy*It)]])

    if np.linalg.det(ATA) > 0.001:
        v = np.linalg.inv(ATA) * ATb 
        new_x, new_y = x + v[0], y + v[1]
        nextPts.append([new_x, new_y])
        return nextPts

cap = cv.VideoCapture("OPTICAL_FLOW.mp4")
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, maxCorners=10, qualityLevel=0.3, minDistance=7).reshape(-1, 2)

while True:
   ret,frame=cap.read()
   frame_gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

while True:
    
    p1 = LKanade(old_gray, frame_gray, p0)
    for old, new in zip(p0, p1):
        x1, y1 = old.astype(int)
        x2, y2 = new.astype(int)
        final= cv.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv.imgShow(final)
    old_gray = frame_gray.copy()
    p0 = p1 

    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv.destroyAllWindows()
    
    
