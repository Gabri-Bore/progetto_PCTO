import pygame
import cv2
import mediapipe as mp
import random
import numpy as np

# Inizializza Pygame
pygame.init()

# Dimensioni della finestra
WIDTH, HEIGHT = 1280, 720

# Colori
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

# Inizializza la finestra
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Hole in the Wall - Body Tracking")

# Clock per il framerate
clock = pygame.time.Clock()

# Mediapipe Pose Detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Funzione per generare una sagoma casuale
def generate_pose():
    return {
        "left_hand": (random.randint(200, 400), random.randint(100, 300)),
        "right_hand": (random.randint(400, 600), random.randint(100, 300)),
        "left_foot": (random.randint(200, 400), random.randint(400, 500)),
        "right_foot": (random.randint(400, 600), random.randint(400, 500)),
        "head": (random.randint(300, 500), random.randint(50, 150))
    }

def draw_silhouette(screen, pose):
    """
    Disegna una sagoma vuota basata sui punti della posa.
    """
    # Estrai i punti della posa
    left_hand = pose["left_hand"]
    right_hand = pose["right_hand"]
    left_foot = pose["left_foot"]
    right_foot = pose["right_foot"]
    head = pose["head"]

    # Crea il contorno come una lista di punti ordinati
    contour_points = [
        left_hand,   # Mano sinistra
        head,        # Testa
        right_hand,  # Mano destra
        right_foot,  # Piede destro
        left_foot    # Piede sinistro
    ]

    # Disegna il poligono chiuso riempito di nero
    pygame.draw.polygon(screen, BLACK, contour_points)

    # (Opzionale) Disegna il bordo del contorno in bianco per evidenziare i limiti
    pygame.draw.polygon(screen, WHITE, contour_points, width=2)

# Funzione per calcolare la distanza tra due punti
def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Funzione per calcolare il punteggio
def calculate_score(player_pose, target_pose):
    score = 0
    for key in target_pose:
        if key in player_pose:
            distance = calculate_distance(player_pose[key], target_pose[key])
            score += max(0, 100 - distance)  # Più vicino, più punti
    return int(score)

# Genera una sagoma iniziale
target_pose = generate_pose()
# Disegna la sagoma
draw_silhouette(screen, target_pose)  

# Stato del gioco
game_running = True
score = 0

# Inizializza la videocamera
cap = cv2.VideoCapture(0)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game_running = False

    if not game_running:
        break

    # Leggi il frame dalla videocamera
    ret, frame = cap.read()
    if not ret:
        print("Errore nell'accesso alla videocamera.")
        break

    # Flip del frame per allinearlo con i movimenti
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Processa il frame con Mediapipe
    results = pose.process(rgb_frame)

    # Ottieni i punti del corpo del giocatore
    player_pose = {}
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        height, width, _ = frame.shape

        # Mappa i punti chiave (esempio: mani, piedi, testa)
        player_pose = {
            "left_hand": (int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x * WIDTH),
                           int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y * HEIGHT)),
            "right_hand": (int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x * WIDTH),
                            int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y * HEIGHT)),
            "left_foot": (int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x * WIDTH),
                           int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y * HEIGHT)),
            "right_foot": (int(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x * WIDTH),
                            int(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y * HEIGHT)),
            "head": (int(landmarks[mp_pose.PoseLandmark.NOSE].x * WIDTH),
                      int(landmarks[mp_pose.PoseLandmark.NOSE].y * HEIGHT))
        }

    # Calcola il punteggio
    if player_pose:
        score = calculate_score(player_pose, target_pose)

    # Disegna sullo schermo
    screen.fill(BLACK)

    # Disegna la sagoma target
    for key, pos in target_pose.items():
        pygame.draw.circle(screen, WHITE, pos, 10)

    # Disegna la posizione del giocatore
    for key, pos in player_pose.items():
        pygame.draw.circle(screen, GREEN, pos, 10)

    # Mostra il punteggio
    font = pygame.font.Font(None, 36)
    score_text = font.render(f"Score: {score}", True, RED)
    screen.blit(score_text, (10, 10))

    # Aggiorna lo schermo
    pygame.display.flip()
    clock.tick(30)

# Rilascia la videocamera e chiudi il gioco
cap.release()
pygame.quit()