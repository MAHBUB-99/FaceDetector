# Import necessary libraries
import cv2
import os

# Define a function to start capturing face images for training
def start_capture(name):
    # Define the path where captured images will be saved
    path = "./data/" + name
    # Initialize the count of captured images
    num_of_images = 0
    # Initialize a face detector using the Haar Cascade classifier
    detector = cv2.CascadeClassifier("./data/haarcascade_frontalface_default.xml")

    try:
        # Try to create a directory for the person's name if it doesn't exist
        os.makedirs(path)
    except:
        # If the directory already exists, print a message
        print('Directory Already Created')

    # Initialize a video capture object (camera)
    vid = cv2.VideoCapture(0)

    # Start an infinite loop for capturing images
    while True:
        # Read a frame from the camera
        ret, img = vid.read()
        new_img = None
        # Convert the frame to grayscale
        grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect faces in the grayscale image
        face = detector.detectMultiScale(image=grayimg, scaleFactor=1.1, minNeighbors=5)

        # Iterate through the detected faces
        for x, y, w, h in face:
            # Draw a rectangle around the detected face
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), 2)
            # Display a text indicating that a face has been detected
            cv2.putText(img, "Face Detected", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))
            # Display the count of captured images
            cv2.putText(img, str(str(num_of_images)+" images captured"), (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))
            # Crop the detected face region
            new_img = img[y:y+h, x:x+w]

        # Display the image with face detection
        cv2.imshow("FaceDetection", img)
        # Wait for a key press (1ms) and mask it with 0xFF to capture the lowest 8 bits
        key = cv2.waitKey(1) & 0xFF

        try:
            # Try to save the captured face image with a unique name
            cv2.imwrite(str(path+"/"+str(num_of_images)+name+".jpg"), new_img)
            num_of_images += 1
        except:
            # If an error occurs while saving, ignore it
            pass

        # Break the loop if 'q' or 'Esc' key is pressed or if the specified number of images is captured
        if key == ord("q") or key == 27 or num_of_images > 310:
            break

    # Close all OpenCV windows and return the count of captured images
    cv2.destroyAllWindows()
    return num_of_images
