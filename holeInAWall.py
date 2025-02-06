import os
import pygame
import sys
import cv2
import mediapipe as mp
import random
import numpy as np
from pygame.locals import *
import time
import csv

# Initialize Pygame
pygame.init()

# Window dimensions
WIDTH, HEIGHT = 1280, 720

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Background
auto = pygame.image.load(os.path.join('images', 'sfondomigliorato.png'))
auto = pygame.transform.scale(auto, (WIDTH, HEIGHT))

# Initialize the window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Hole in the Wall")

# Clock for the framerate
clock = pygame.time.Clock()

# Mediapipe Pose Detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Time threshold variables
initial_time_threshold = 10  # 10 seconds initially
time_threshold = initial_time_threshold
start_time = time.time()

# Font for displaying the remaining time
font = pygame.font.Font(None, 74)

# Variable to check if the game is ready
game_ready = False

# Variable to store the selected body part
selected_body_part = None  # None means all body parts are active

# Menu options
menu_options = ["Corpo", "Braccia", "Gambe", "Testa"]
current_option = 0

def salva_punteggio(punteggio):
    file_path = "punteggio.csv"

    # Controlla se esiste un punteggio precedente
    ultimo_punteggio = 0
    ultimi_tre = []
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            righe = file.readlines()
            ultimi_tre = [int(r.strip().replace("%", "")) for r in righe[-3:] if r.strip().replace("%", "").isdigit()]
            if righe:
                try:
                    ultimo_punteggio = int(righe[-1].strip().replace("%", ""))  # Prende l'ultimo punteggio salvato
                except ValueError:
                    ultimo_punteggio = 0

    # Salva il punteggio con il simbolo %
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([f"{punteggio}%"])

    print(f"Punteggio {punteggio}% salvato correttamente!")
    

def draw_feedback_points(screen, player_pose, silhouette, threshold=50):
    for key, (sx, sy) in silhouette.items():
        if key in player_pose:
            px, py = player_pose[key]
            distanza = np.sqrt((px - sx) ** 2 + (py - sy) ** 2)
            # Se entro la soglia, colore verde; altrimenti rosso (o un altro colore)
            color = GREEN if distanza <= threshold else RED
            # Disegna un piccolo cerchio di feedback
            pygame.draw.circle(screen, color, (sx, sy), 10)


def calcola_punteggio(player_pose, silhouette):
    if not player_pose:
        return 0  # Se non ci sono dati del giocatore, il punteggio è 0

    punteggio_totale = 0
    punti_confronto = 0

    for key in silhouette.keys():
        if key in player_pose:
            # Calcola la distanza tra il punto del giocatore e quello della silhouette
            px, py = player_pose[key]
            sx, sy = silhouette[key]
            distanza = np.sqrt((px - sx) ** 2 + (py - sy) ** 2)
            
            # Normalizziamo il punteggio tra 0 e 100 
            punteggio = max(0, 100 - (distanza / 2))
            
            punteggio_totale += punteggio
            punti_confronto += 1

    if punti_confronto == 0:
        return 0  # Evita divisioni per zero

    return int(punteggio_totale / punti_confronto)  # Media dei punteggi

def draw_menu():
    screen.fill(BLACK)
    for i, option in enumerate(menu_options):
        color = GREEN if i == current_option else WHITE
        text = font.render(option, True, color)
        screen.blit(text, (WIDTH // 2 - text.get_width() // 2, HEIGHT // 2 - 100 + i * 100))
    pygame.display.flip()

def generate_random_silhouette():
    center_x, center_y = WIDTH // 2, HEIGHT // 2
    
    # Definiamo angoli naturali per le articolazioni
    shoulder_angle = random.uniform(-30, 30)  # Angolo delle spalle rispetto all'orizzontale
    elbow_angle = random.uniform(0, 90)      # Angolo dei gomiti (evita posizioni innaturali)
    hip_angle = random.uniform(-20, 20)      # Angolo dei fianchi
    knee_angle = random.uniform(0, 60)       # Angolo delle ginocchia
    
    # Lunghezze degli arti aumentate del 50% (proporzioni mantenute)
    arm_length = 120      # Era 80
    forearm_length = 105  # Era 70
    leg_length = 150      # Era 100
    shin_length = 135     # Era 90
    
    # Calcolo posizione testa (più in alto e più grande)
    head = (center_x + random.randint(-30, 30), 
            center_y - 225 + random.randint(-15, 15))  # Era -150
    
    # Calcolo spalle usando l'angolo (aumentata la distanza tra le spalle)
    shoulder_base_y = head[1] + 75  # Era 50
    left_shoulder = (center_x - 75 * np.cos(np.radians(shoulder_angle)),  # Era 50
                    shoulder_base_y + 75 * np.sin(np.radians(shoulder_angle)))
    right_shoulder = (center_x + 75 * np.cos(np.radians(shoulder_angle)),
                     shoulder_base_y + 75 * np.sin(np.radians(shoulder_angle)))
    
    # Calcolo gomiti usando l'angolo
    left_elbow = (left_shoulder[0] - arm_length * np.cos(np.radians(elbow_angle)),
                 left_shoulder[1] + arm_length * np.sin(np.radians(elbow_angle)))
    right_elbow = (right_shoulder[0] + arm_length * np.cos(np.radians(elbow_angle)),
                  right_shoulder[1] + arm_length * np.sin(np.radians(elbow_angle)))
    
    # Calcolo mani (estensione naturale dei gomiti)
    left_hand = (left_elbow[0] - forearm_length * np.cos(np.radians(elbow_angle - 20)),
                left_elbow[1] + forearm_length * np.sin(np.radians(elbow_angle - 20)))
    right_hand = (right_elbow[0] + forearm_length * np.cos(np.radians(elbow_angle - 20)),
                 right_elbow[1] + forearm_length * np.sin(np.radians(elbow_angle - 20)))
    
    # Calcolo fianchi (aumentata la distanza tra i fianchi e dalle spalle)
    hip_base_y = shoulder_base_y + 180  # Era 120
    left_hip = (center_x - 60 * np.cos(np.radians(hip_angle)),  # Era 40
                hip_base_y + 60 * np.sin(np.radians(hip_angle)))
    right_hip = (center_x + 60 * np.cos(np.radians(hip_angle)),
                 hip_base_y + 60 * np.sin(np.radians(hip_angle)))
    
    # Calcolo ginocchia
    left_knee = (left_hip[0] - leg_length * np.sin(np.radians(knee_angle)),
                left_hip[1] + leg_length * np.cos(np.radians(knee_angle)))
    right_knee = (right_hip[0] + leg_length * np.sin(np.radians(knee_angle)),
                 right_hip[1] + leg_length * np.cos(np.radians(knee_angle)))
    
    # Calcolo piedi
    left_foot = (left_knee[0], left_knee[1] + shin_length)
    right_foot = (right_knee[0], right_knee[1] + shin_length)
    
    return {
        "head": head,
        "left_shoulder": right_shoulder,
        "right_shoulder": left_shoulder,
        "left_elbow": right_elbow,
        "right_elbow": left_elbow,
        "left_hand": right_hand,
        "right_hand": left_hand,
        "left_hip": right_hip,
        "right_hip": left_hip,
        "left_knee": right_knee,
        "right_knee": left_knee,
        "left_foot": right_foot,
        "right_foot": left_foot
    }

def draw_silhouette(screen, pose, color=WHITE):
    if not pose:
        return
    
    # Verifica la presenza dei punti chiave
    if not all(key in pose for key in ["head", "left_shoulder", "right_shoulder"]):
        return

    # Punti base
    head = pose["head"]
    left_shoulder = pose["left_shoulder"]
    right_shoulder = pose["right_shoulder"]
    
    # Calcolo punti anatomici derivati
    neck = ((left_shoulder[0] + right_shoulder[0]) // 2, 
            (left_shoulder[1] + right_shoulder[1]) // 2)
    
    # Calcolo della larghezza delle spalle per proporzioni
    shoulder_width = np.sqrt((left_shoulder[0] - right_shoulder[0])**2 + 
                           (left_shoulder[1] - right_shoulder[1])**2)
    head_radius = int(shoulder_width * 0.3)
    
    # Spessore linee aumentato
    line_thickness = int(head_radius * 0.7)  # Aumentato lo spessore base
    joint_radius = int(head_radius * 0.4)    # Aumentato il raggio delle articolazioni
    
    # Disegno struttura base con linee più spesse
    pygame.draw.circle(screen, color, head, head_radius + line_thickness//2.25)
    pygame.draw.line(screen, color, head, neck, line_thickness)
    
    # Spalle e torso
    if left_shoulder and right_shoulder:
        pygame.draw.line(screen, color, left_shoulder, right_shoulder, line_thickness)
        pygame.draw.circle(screen, color, left_shoulder, joint_radius)
        pygame.draw.circle(screen, color, right_shoulder, joint_radius)
    
    # Braccia con articolazioni più evidenti
    if "left_elbow" in pose and "left_hand" in pose:
        left_elbow = pose["left_elbow"]
        left_hand = pose["left_hand"]
        
        pygame.draw.line(screen, color, left_shoulder, left_elbow, line_thickness)
        pygame.draw.line(screen, color, left_elbow, left_hand, line_thickness)
        pygame.draw.circle(screen, color, left_elbow, joint_radius)
        pygame.draw.circle(screen, color, left_hand, joint_radius)
    
    if "right_elbow" in pose and "right_hand" in pose:
        right_elbow = pose["right_elbow"]
        right_hand = pose["right_hand"]
        
        pygame.draw.line(screen, color, right_shoulder, right_elbow, line_thickness)
        pygame.draw.line(screen, color, right_elbow, right_hand, line_thickness)
        pygame.draw.circle(screen, color, right_elbow, joint_radius)
        pygame.draw.circle(screen, color, right_hand, joint_radius)
    
    # Torso più definito
    if "left_hip" in pose and "right_hip" in pose:
        left_hip = pose["left_hip"]
        right_hip = pose["right_hip"]
        
        # Torso con spessore aumentato
        torso_points = [
            left_shoulder,
            right_shoulder,
            right_hip,
            left_hip
        ]
        pygame.draw.polygon(screen, color, torso_points)
        
        # Articolazioni dei fianchi più evidenti
        pygame.draw.circle(screen, color, left_hip, joint_radius)
        pygame.draw.circle(screen, color, right_hip, joint_radius)
    
    # Gambe con spessore aumentato
    if "left_knee" in pose and "left_foot" in pose:
        left_knee = pose["left_knee"]
        left_foot = pose["left_foot"]
        
        pygame.draw.line(screen, color, left_hip, left_knee, line_thickness)
        pygame.draw.line(screen, color, left_knee, left_foot, line_thickness)
        pygame.draw.circle(screen, color, left_knee, joint_radius)
        pygame.draw.circle(screen, color, left_foot, joint_radius)
    
    if "right_knee" in pose and "right_foot" in pose:
        right_knee = pose["right_knee"]
        right_foot = pose["right_foot"]
        
        pygame.draw.line(screen, color, right_hip, right_knee, line_thickness)
        pygame.draw.line(screen, color, right_knee, right_foot, line_thickness)
        pygame.draw.circle(screen, color, right_knee, joint_radius)
        pygame.draw.circle(screen, color, right_foot, joint_radius)

# Initialize the camera
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
FRAME_HEIGHT, FRAME_WIDTH = frame.shape[:2]

# Generate a random silhouette with limited movements
random_silhouette = generate_random_silhouette()

# Menu loop
menu_running = True
while menu_running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            menu_running = False
            game_running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                current_option = (current_option - 1) % len(menu_options)
            elif event.key == pygame.K_DOWN:
                current_option = (current_option + 1) % len(menu_options)
            elif event.key == pygame.K_RETURN:
                selected_body_part = menu_options[current_option]
                menu_running = False
                game_running = True
                start_time = time.time()

    draw_menu()

# Main game loop
while game_running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game_running = False

    # Read the frame from the camera
    ret, frame = cap.read()
    if not ret:
        print("Error accessing the camera.")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    # Get the body pose points
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

    #Clear the screen and draw the background
    screen.blit(auto, (0, 0))

    # Draw the player's pose if detected, otherwise draw the random silhouette
    if player_pose:
        # Disegna la silhouette del giocatore (in nero) e quella target (in bianco)
        draw_silhouette(screen, player_pose, BLACK)
        draw_silhouette(screen, random_silhouette, WHITE)
    
    # Aggiungi i feedback colorati sui punti
    draw_feedback_points(screen, player_pose, random_silhouette, threshold=30)
    
    # Calcola il punteggio, etc...
    punteggio = calcola_punteggio(player_pose, random_silhouette)

    # Set game_ready to True after the first frame is displayed
    if not game_ready:
        game_ready = True
        start_time = time.time()

    # Check if the time threshold has been exceeded
    if game_ready:
        elapsed_time = time.time() - start_time
        remaining_time = max(0, time_threshold - elapsed_time)

        # Render the remaining time on the screen
        time_text = font.render(f"Tempo: {int(remaining_time)}", True, GREEN)
        screen.blit(time_text, (10, 10))

        if elapsed_time > time_threshold:
            print(f"Punteggio finale: {punteggio}")  # Debug
            salva_punteggio(punteggio)  # Salva il punteggio nel CSV

            # Update only the selected body part in the random silhouette
            if selected_body_part == "Gambe e braccia":
                random_silhouette = generate_random_silhouette()
            elif selected_body_part == "Braccia":
                random_silhouette["left_elbow"] = (random_silhouette["left_elbow"][0] + random.randint(-20, 20),
                                                   random_silhouette["left_elbow"][1] + random.randint(-20, 20))
                random_silhouette["right_elbow"] = (random_silhouette["right_elbow"][0] + random.randint(-20, 20),
                                                    random_silhouette["right_elbow"][1] + random.randint(-20, 20))
                random_silhouette["left_hand"] = (random_silhouette["left_hand"][0] + random.randint(-20, 20),
                                                  random_silhouette["left_hand"][1] + random.randint(-20, 20))
                random_silhouette["right_hand"] = (random_silhouette["right_hand"][0] + random.randint(-20, 20),
                                                   random_silhouette["right_hand"][1] + random.randint(-20, 20))
            elif selected_body_part == "Gambe":
                random_silhouette["left_knee"] = (random_silhouette["left_knee"][0] + random.randint(-20, 20),
                                                  random_silhouette["left_knee"][1] + random.randint(-20, 20))
                random_silhouette["right_knee"] = (random_silhouette["right_knee"][0] + random.randint(-20, 20),
                                                   random_silhouette["right_knee"][1] + random.randint(-20, 20))
                random_silhouette["left_foot"] = (random_silhouette["left_foot"][0] + random.randint(-20, 20),
                                                  random_silhouette["left_foot"][1] + random.randint(-20, 20))
                random_silhouette["right_foot"] = (random_silhouette["right_foot"][0] + random.randint(-20, 20),
                                                   random_silhouette["right_foot"][1] + random.randint(-20, 20))
            elif selected_body_part == "Testa":
                random_silhouette["head"] = (random_silhouette["head"][0] + random.randint(-20, 20),
                                             random_silhouette["head"][1] + random.randint(-20, 20))

            start_time = time.time()
            time_threshold = max(10, time_threshold - 0.5)

    # Update the display
    pygame.display.update()
    pygame.display.flip()
    clock.tick(60)

cap.release()
pygame.quit()