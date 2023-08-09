import cv2
import face_recognition
import requests

# Azure Face API endpoint and subscription key
FACE_API_ENDPOINT = "YOUR_FACE_API_ENDPOINT"
SUBSCRIPTION_KEY = "YOUR_SUBSCRIPTION_KEY"

def get_age_gender(face_image):
    url = f"{FACE_API_ENDPOINT}/detect"
    headers = {
        "Ocp-Apim-Subscription-Key": SUBSCRIPTION_KEY,
        "Content-Type": "application/octet-stream",
    }

    response = requests.post(url, headers=headers, data=face_image)
    data = response.json()

    if data and "faceAttributes" in data[0]:
        attributes = data[0]["faceAttributes"]
        gender = attributes["gender"]
        age = attributes["age"]
        return gender, age
    else:
        return None, None

def main():
    # Load an example image and learn how to recognize it
    known_image = face_recognition.load_image_file("known_face.jpg")
    known_face_encoding = face_recognition.face_encodings(known_image)[0]

    # Open a camera feed
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()

        # Find all face locations and encodings in the current frame
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Compare face encoding with known face encoding
            matches = face_recognition.compare_faces([known_face_encoding], face_encoding)
            name = "Unknown"
            gender = "Unknown"
            age = "Unknown"

            if True in matches:
                name = "Known Person"
                face_image = cv2.imencode('.jpg', frame[top:bottom, left:right])[1].tobytes()
                gender, age = get_age_gender(face_image)

            # Draw rectangle around the face and display the result
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            label = f"{name}\nGender: {gender}\nAge: {age}"
            cv2.putText(frame, label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Display the frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close the window
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
