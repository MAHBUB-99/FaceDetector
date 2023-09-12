# Import the OpenCV library for computer vision tasks
import cv2
# Import the sleep function from the time module
from time import sleep
# Import the Image class from the PIL (Pillow) library for image processing
from PIL import Image

# Define a function named main_app that takes a 'name' argument


def main_app(name):

    # Initialize a Haar Cascade Classifier for face detection
    face_cascade = cv2.CascadeClassifier(
        './data/haarcascade_frontalface_default.xml')
    # Create an LBPH (Local Binary Pattern Histograms) face recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    classifier_file_path = f"./data/classifiers/{name}_classifier.xml"
    # line for debugging
    print(f"Classifier file path: {classifier_file_path}")

    # Read the trained face recognizer model specific to 'name'
    recognizer.read(classifier_file_path)
    # Initialize a video capture object to access the default camera (index 0)
    cap = cv2.VideoCapture(0)
    # Initialize a prediction variable
    pred = 0

    # Start an infinite loop for video processing
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Iterate through detected faces
        for (x, y, w, h) in faces:
            # Extract the region of interest (ROI) for the detected face
            roi_gray = gray[y:y+h, x:x+w]

            # Predict the ID and confidence of the detected face
            id, confidence = recognizer.predict(roi_gray)
            # Invert and store the confidence score
            confidence = 100 - int(confidence)
            # Reset the prediction variable
            pred = 0

            # Check if confidence is above a threshold
            if confidence > 50:
                # Increment the prediction variable
                pred += +1
                # Set text and font for recognized face
                text = name.upper()
                font = cv2.FONT_HERSHEY_PLAIN
                # Draw a green rectangle around the face
                frame = cv2.rectangle(
                    frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Display the recognized name on the frame
                frame = cv2.putText(frame, text, (x, y-4),
                                    font, 1, (0, 255, 0), 1, cv2.LINE_AA)
            else:
                # Decrement the prediction variable
                pred += -1
                # Set text and font for unrecognized face
                text = "UnknownFace"
                font = cv2.FONT_HERSHEY_PLAIN
                # Draw a red rectangle around the face
                frame = cv2.rectangle(
                    frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                # Display "UnknownFace" on the frame
                frame = cv2.putText(frame, text, (x, y-4),
                                    font, 1, (0, 0, 255), 1, cv2.LINE_AA)

        # Show the processed frame with detected faces
        cv2.imshow("image", frame)

        # Check if the 'q' key is pressed
        if cv2.waitKey(20) & 0xFF == ord('q'):
            # Print the value of 'pred'
            print(pred)
            # Check if 'pred' is positive
            if pred > 0:
                # Define dimensions for resizing images
                dim = (124, 124)
                # Read an image based on 'pred' and 'name' parameters
                img = cv2.imread(
                    f".\\data\\{name}\\{pred}{name}.jpg", cv2.IMREAD_UNCHANGED)
                # Resize the image
                resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
                # Save the resized image with a new name
                cv2.imwrite(f".\\data\\{name}\\50{name}.jpg", resized)
                # Open an image named "2.png"
                Image1 = Image.open(f".\\2.png")
                # Make a copy of the image
                Image1copy = Image1.copy()
                # Open the resized image
                Image2 = Image.open(f".\\data\\{name}\\50{name}.jpg")
                # Make a copy of the resized image
                Image2copy = Image2.copy()
                # Paste the resized image onto the original image at a specific position
                Image1copy.paste(Image2copy, (195, 114))
                # Save the final image as "end.png"
                Image1copy.save("end.png")
                # Read and display the final image
                frame = cv2.imread("end.png", 1)
                cv2.imshow("Result", frame)
                cv2.waitKey(5000)
            # Break out of the loop
            break

    # Release the video capture object
    cap.release()
    # Close all OpenCV windows
    cv2.destroyAllWindows()
