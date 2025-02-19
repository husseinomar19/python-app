import cv2

# Laad de ingebouwde gezicht-detector (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Open de webcam (gebruik 0 voor de standaardcamera)
cap = cv2.VideoCapture(0)

while True:
    # Lees een frame van de webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Converteer het frame naar grijswaarden
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecteer gezichten
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Teken rechthoeken rond gezichten
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Toon het resultaat
    cv2.imshow("Gezichtsdetectie", frame)

    # Druk op 'q' om te stoppen
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Sluit alles af
cap.release()
cv2.destroyAllWindows()
