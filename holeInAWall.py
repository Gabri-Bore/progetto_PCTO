import os
import pygame
import sys
import cv2
import mediapipe as mp
import random
import numpy as np
from pygame.locals import *

# Inizializza Pygame
pygame.init()

# Dimensioni della finestra
WIDTH, HEIGHT = 1280, 720

# Colori
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Sfondo
wall = pygame.image.load(os.path.join('images', 'wall.png'))
wall = pygame.transform.scale(wall, (WIDTH, HEIGHT))

# Inizializza la finestra
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Hole in the Wall")

# Clock per il framerate
clock = pygame.time.Clock()

# Mediapipe Pose Detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def draw_silhouette(screen, pose):
    if not pose:
        return
    
    # Estrarre i punti chiave
    head = pose["head"]
    neck = ((head[0], head[1] + 40))
    left_shoulder = pose["left_shoulder"]
    right_shoulder = pose["right_shoulder"]
    left_elbow = pose["left_elbow"]
    right_elbow = pose["right_elbow"]
    left_hand = pose["left_hand"]
    right_hand = pose["right_hand"]
    left_hip = pose["left_hip"]
    right_hip = pose["right_hip"]
    left_knee = pose["left_knee"]
    right_knee = pose["right_knee"]
    left_foot = pose["left_foot"]
    right_foot = pose["right_foot"]

    # Disegna la testa più grande e una linea singola che la collega al corpo
    pygame.draw.circle(screen, BLACK, head, 90)  # Testa più grande
    pygame.draw.line(screen, BLACK, head, neck, 20)  # Collega la testa al corpo
    pygame.draw.line(screen, BLACK, neck, left_shoulder, 20)
    pygame.draw.line(screen, BLACK, neck, right_shoulder, 20)
    pygame.draw.line(screen, BLACK, left_shoulder, right_shoulder, 20)
    pygame.draw.line(screen, BLACK, left_shoulder, left_elbow, 20)
    pygame.draw.line(screen, BLACK, left_elbow, left_hand, 20)
    pygame.draw.line(screen, BLACK, right_shoulder, right_elbow, 20)
    pygame.draw.line(screen, BLACK, right_elbow, right_hand, 20)
    pygame.draw.line(screen, BLACK, left_shoulder, left_hip, 20)
    pygame.draw.line(screen, BLACK, right_shoulder, right_hip, 20)
    pygame.draw.line(screen, BLACK, left_hip, right_hip, 20)
    pygame.draw.line(screen, BLACK, left_hip, left_knee, 20)
    pygame.draw.line(screen, BLACK, left_knee, left_foot, 20)
    pygame.draw.line(screen, BLACK, right_hip, right_knee, 20)
    pygame.draw.line(screen, BLACK, right_knee, right_foot, 20)

# Inizializza la videocamera
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
FRAME_HEIGHT, FRAME_WIDTH = frame.shape[:2]

game_running = True
while game_running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game_running = False

    # Leggi il frame dalla videocamera
    ret, frame = cap.read()
    if not ret:
        print("Errore nell'accesso alla videocamera.")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    # Ottieni i punti del corpo del giocatore
    player_pose = {}
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        player_pose = {
            "head": (int(landmarks[mp_pose.PoseLandmark.NOSE].x * WIDTH),
                      int(landmarks[mp_pose.PoseLandmark.NOSE].y * HEIGHT - 50)),
            "left_shoulder": (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * WIDTH),
                               int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * HEIGHT)),
            "right_shoulder": (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * WIDTH),
                                int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * HEIGHT)),
            "left_elbow": (int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x * WIDTH),
                            int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y * HEIGHT)),
            "right_elbow": (int(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x * WIDTH),
                             int(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y * HEIGHT)),
            "left_hand": (int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x * WIDTH),
                           int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y * HEIGHT)),
            "right_hand": (int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x * WIDTH),
                            int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y * HEIGHT)),
            "left_hip": (int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * WIDTH),
                          int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * HEIGHT)),
            "right_hip": (int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x * WIDTH),
                           int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y * HEIGHT)),
            "left_knee": (int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x * WIDTH),
                           int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y * HEIGHT)),
            "right_knee": (int(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x * WIDTH),
                            int(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y * HEIGHT)),
            "left_foot": (int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x * WIDTH),
                           int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y * HEIGHT)),
            "right_foot": (int(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x * WIDTH),
                            int(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y * HEIGHT))
        }

    screen.blit(auto, (0, 0))
    if player_pose:
        score = calculate_score(player_pose, target_pose)

    # Incrementa il timer per aumentare la difficoltà
    time_elapsed += clock.get_time()
    if time_elapsed > difficulty_timer:
        target_pose = generate_pose()
        time_elapsed = 0

    # Disegna sullo schermo
    screen.blit(wall, (0, 0))  # Mostra lo sfondo

    # Disegna la sagoma target
    for key, pos in target_pose.items():
        pygame.draw.circle(screen, BLUE, pos, 10)

    # Disegna la sagoma del giocatore
    if results.pose_landmarks:
        for landmark in player_pose.values():
            pygame.draw.circle(screen, RED, landmark, 10)

    # Mostra il punteggio
    font = pygame.font.Font(None, 36)
    score_text = font.render(f"Punteggio: {score}", True, RED)
    screen.blit(score_text, (10, 10))

    # Aggiorna lo schermo
    pygame.display.flip()
    clock.tick(30)

cap.release()
pygame.quit()