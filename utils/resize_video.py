import cv2
import numpy as np


# Create a VideoCapture object
cap = cv2.VideoCapture(r"C:\Users\user\Downloads\full_vids\20_3rd.avi")

# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
counter = 0
while (True):
    ret, frame = cap.read()
    if ret == True:

        # Write the frame into the file 'output.avi'
        dim = (int(frame_width * 0.5), int(frame_height * 0.5))
        #print(frame_width, frame_height)
        #print(dim)
        # resize image
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        cv2.imwrite(f'resized_images/particle_6_{counter}.jpg', frame)
        counter += 1
        # Display the resulting frame
        cv2.imshow('frame', frame)

        # Press Q on keyboard to stop recording
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

    # When everything done, release the video capture and video write objects
cap.release()


# Closes all the frames
cv2.destroyAllWindows()

