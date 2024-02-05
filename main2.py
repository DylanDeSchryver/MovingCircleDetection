#THIS VERSION DETECTS THE OUTSIDE OF A MASK FROM THE ORIGINAL CIRCLE DETECTED ON EACH FRAME

import cv2
import numpy as np


cap = cv2.VideoCapture('IMG_7344.mov')  #video

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

        # Use the parameters of the first detected circle to create a mask
        mask = np.zeros_like(gray)
        cv2.circle(mask, (circles[0, 0, 0], circles[0, 0, 1]), circles[0, 0, 2], 255, -1)

        # Invert the mask to exclude the region inside the original circle
        mask = cv2.bitwise_not(mask)

        # Apply the mask to the grayscale image
        gray_outside = cv2.bitwise_and(gray, gray, mask=mask)

        # Use HoughCircles again on the modified grayscale image to find circles outside the original circle
        outer_circles = cv2.HoughCircles(
            gray_outside, cv2.HOUGH_GRADIENT, dp=1, minDist=999)

        if outer_circles is not None: #if circles are found
            # Extract the outer circle parameters
            outer_circles = np.uint16(np.around(outer_circles))

            for i in outer_circles[0, :]:
                # Draw the outer circle on the frame for outer circles
                cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 5)

                # Draw the center of the circle for outer circles
                cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)

    # Display the resulting frame
    cv2.imshow("Circle", frame)
    cv2.imshow('Mask', gray_outside)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
