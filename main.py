import mss  # mss (Multiple ScreenShots) is an ultra-fast, cross-platform library used to capture screenshots
import cv2
import numpy as np

fps = 10
update_interval = int(1000/fps) #thousand mili second (1 sec) / fps , delay must be in int
    
# pre-create window ONCE before the loop with a unique handle
cv2.namedWindow("Screen Capture", cv2.WINDOW_NORMAL)
    
with mss.mss() as sct:
    monitor = sct.monitors[1]
    
    while True:
        screenshot = sct.grab(monitor)
        img=np.array(screenshot) #this will convert the screenshot to numpy array

         #convert image form BGRA Blue, Green, Red, and Alpha/opacity (mss format) to BRG (opencv format)
        img_bgr = cv2.cvtColor(img,cv2.COLOR_BGRA2BGR)
        cv2.imshow("Screen Capture",img_bgr)

        key = cv2.waitKey(update_interval) & 0xFF
        if key == ord('q'): #program will wait for some time and if q is not pressed next loop will continue
            break

cv2.destroyAllWindows()