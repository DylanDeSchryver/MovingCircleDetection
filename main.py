
import cv2
import numpy as np


cap = cv2.VideoCapture('IMG_7344.MOV')  #video

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and improve circle detection
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)

    # Use HoughCircles to detect circles in the image
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=999)

    if circles is not None: #if circles are found
        # Extract the circle parameters
        circles = np.uint16(np.around(circles))

        for i in circles[0, :]:
            # Draw the outer circle on the frame
            cv2.circle(frame, (i[0], i[1]), i[2], (255, 0, 255), 5)

            # Draw the center of the circle
            cv2.circle(frame, (i[0], i[1]), 2, (255, 0, 255), 3)

    # Display the resulting frame
    cv2.imshow("Circle", frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()