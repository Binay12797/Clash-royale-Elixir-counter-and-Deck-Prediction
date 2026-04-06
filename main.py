import mss  # mss (Multiple ScreenShots) is an ultra-fast, cross-platform library used to capture screenshots
import cv2
import numpy as np

with mss.mss() as sct:
    monitor = sct.monitors[1]
    screenshot = sct.grab(monitor)
    img=np.array(screenshot) #this will convert the screenshot to numpy array

    #convert image form BGRA Blue, Green, Red, and Alpha/opacity (mss format) to BRG (opencv format)
    img_bgr = cv2.cvtColor(img,cv2.COLOR_BGRA2BGR)
    #cv2.imshow("Screen Capture", img_bgr)
    cv2.imshow("Screen Capture", img)

    cv2.waitKey(0) #wait till an random key is pressed
    cv2.destroyAllWindows()