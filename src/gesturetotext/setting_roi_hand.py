import cv2
import numpy as np
import pickle

# captures the pixel value of the small square and turns it into numpy array
def build_squares(img):
    x, y, w, h = 420, 140, 10, 10  
    d = 10
    imgCrop = None
    crop = None
    
    for i in range(10):
        for j in range(6):
            if np.any(imgCrop == None):
                imgCrop = img[y:y+h, x:x+w]
            else:
                imgCrop = np.hstack((imgCrop, img[y:y+h, x:x+w]))
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,255), 1)
            x+=w+d
        if np.any(crop == None):
            crop = imgCrop
        else:
            crop = np.vstack((crop, imgCrop)) 
        imgCrop = None
        x = 420
        y+=h+d
        
    crop1 = crop[0:(crop.shape[0]),int(crop.shape[1]/6):]
    return crop1
    

def get_hand_hist():
    cam = cv2.VideoCapture(0)
    x, y, w, h = 420, 140, 110, 190
    flagPressedC, flagPressedS = False, False
    imgCrop = None
    
    while True:
        img = cam.read()[1]
        img = cv2.flip(img, 1)
        img = cv2.resize(img, (640, 480))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        keypress = cv2.waitKey(1) & 0xFF

        if keypress == ord('h'):        
            hsvCrop = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2HSV)
            flagPressedC = True
            hist = cv2.calcHist([hsvCrop], [0, 1], None, [180, 256], [0, 180, 0, 256])
            cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
        
        elif keypress == ord('s'):
            flagPressedS = True 
            break
        
        if flagPressedC:    
            dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
            dst1 = dst.copy()
            disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
            cv2.filter2D(dst,-1,disc,dst)
            
            kernel1 = np.ones ((8,8),np.uint8)
            erosion = cv2.erode(dst, kernel1, iterations =1)
            kernel2 = np.ones((12,12),np.uint8)
            dst = cv2.dilate(erosion ,kernel2,iterations = 1)

            blur = cv2.GaussianBlur(dst, (11,11), 0)
            blur = cv2.medianBlur(blur, 15)
            ret,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            thresh = cv2.merge((thresh,thresh,thresh))
            cv2.imshow("Thresh", thresh)
        
        if not flagPressedS:
            imgCrop = build_squares(img)
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,255), 2)
        cv2.imshow("Set hand histogram", img)
    
    cam.release()
    cv2.destroyAllWindows()
   
    with open("hist", "wb") as f:
        pickle.dump(hist, f)


if __name__ == "__main__":
    get_hand_hist()
# get_hand_hist()