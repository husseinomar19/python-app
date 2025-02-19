import face_recognition
import cv2
import numpy as np

# Laad bekende gezichten
known_image = face_recognition.load_image_file("known_face.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Converteer frame naar RGB
    rgb_frame = frame[:, :, ::-1]

    # Detecteer gezichten in het huidige frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Vergelijk het gezicht met het bekende gezicht
        matches = face_recognition.compare_faces([known_encoding], face_encoding)
        name = "Onbekend"

        if True in matches:
            name = "Bekend Persoon"

        # Teken rechthoek om gezicht en label
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Toon de video
    cv2.imshow("Gezichtsherkenning", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
