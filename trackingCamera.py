import cv2
import mediapipe as mp

# Inizializza MediaPipe Pose e i disegni utili
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Crea l'oggetto Pose
pose = mp_pose.Pose()

# Apri la webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Errore nel leggere il frame dalla webcam")
        break

    # Converti l'immagine in RGB per MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Esegui il rilevamento della posa
    results = pose.process(rgb_frame)

    # Disegna i keypoint e gli scheletri sull'immagine
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
        )

    # Mostra l'immagine con le pose rilevate
    cv2.imshow('Pose Detection', frame)

    # Premi 'q' per uscire
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Rilascia la videocamera e chiudi le finestre
cap.release()
cv2.destroyAllWindows()
