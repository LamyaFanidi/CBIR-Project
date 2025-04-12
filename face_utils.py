import cv2
import numpy as np
import face_recognition
import tempfile


def capture_face(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(file.read())
        tmp_path = tmp_file.name

    image = face_recognition.load_image_file(tmp_path)
    encodings = face_recognition.face_encodings(image)

    if not encodings:
        raise ValueError("Aucun visage détecté.")
    return encodings[0]


def capture_face_live():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Appuyez sur 'A' pour capturer votre visage", frame)

        if cv2.waitKey(1) & 0xFF == ord('a'):
            cap.release()
            cv2.destroyAllWindows()

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb_frame)

            if encodings:
                return encodings[0]
            else:
                raise ValueError("Aucun visage détecté.")
    cap.release()
    cv2.destroyAllWindows()
    return None
