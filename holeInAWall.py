import os
import pygame
import sys
import cv2
import mediapipe as mp
import random
import numpy as np
from pygame.locals import *
import time

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
initial_time_threshold = 20  # 20 seconds initially
time_threshold = initial_time_threshold
start_time = time.time()

# Font for displaying the remaining time
font = pygame.font.Font(None, 74)

# Variable to check if the game is ready
game_ready = False

# Variable to store the selected body part
selected_body_part = None  # None means all body parts are active

# Menu options
menu_options = ["Gambe e braccia", "Braccia", "Gambe", "Testa"]
current_option = 0

def draw_menu():
    screen.fill(BLACK)
    for i, option in enumerate(menu_options):
        color = GREEN if i == current_option else WHITE
        text = font.render(option, True, color)
        screen.blit(text, (WIDTH // 2 - text.get_width() // 2, HEIGHT // 2 - 100 + i * 100))
    pygame.display.flip()

def generate_limited_silhouette():
    center_x, center_y = WIDTH // 2, HEIGHT // 2
    offset_range = 20

    head = (center_x + random.randint(-offset_range, offset_range), 
            center_y - 150 + random.randint(-offset_range, offset_range))
    
    left_shoulder = (center_x - 50 + random.randint(-offset_range, offset_range), 
                     center_y - 50 + random.randint(-offset_range, offset_range))
    right_shoulder = (center_x + 50 + random.randint(-offset_range, offset_range), 
                      center_y - 50 + random.randint(-offset_range, offset_range))
    
    left_elbow = (center_x - 100 + random.randint(-offset_range, offset_range), 
                  center_y + 50 + random.randint(-offset_range, offset_range))
    right_elbow = (center_x + 100 + random.randint(-offset_range, offset_range), 
                   center_y + 50 + random.randint(-offset_range, offset_range))
    
    left_hand = (center_x - 150 + random.randint(-offset_range, offset_range), 
                 center_y + 150 + random.randint(-offset_range, offset_range))
    right_hand = (center_x + 150 + random.randint(-offset_range, offset_range), 
                  center_y + 150 + random.randint(-offset_range, offset_range))
    
    left_hip = (center_x - 50 + random.randint(-offset_range, offset_range), 
                center_y + 100 + random.randint(-offset_range, offset_range))
    right_hip = (center_x + 50 + random.randint(-offset_range, offset_range), 
                 center_y + 100 + random.randint(-offset_range, offset_range))
    
    left_knee = (center_x - 50 + random.randint(-offset_range, offset_range), 
                 center_y + 200 + random.randint(-offset_range, offset_range))
    right_knee = (center_x + 50 + random.randint(-offset_range, offset_range), 
                  center_y + 200 + random.randint(-offset_range, offset_range))
    
    left_foot = (center_x - 50 + random.randint(-offset_range, offset_range), 
                 center_y + 300 + random.randint(-offset_range, offset_range))
    right_foot = (center_x + 50 + random.randint(-offset_range, offset_range), 
                  center_y + 300 + random.randint(-offset_range, offset_range))
    
    return {
        "head": head,
        "left_shoulder": left_shoulder,
        "right_shoulder": right_shoulder,
        "left_elbow": left_elbow,
        "right_elbow": right_elbow,
        "left_hand": left_hand,
        "right_hand": right_hand,
        "left_hip": left_hip,
        "right_hip": right_hip,
        "left_knee": left_knee,
        "right_knee": right_knee,
        "left_foot": left_foot,
        "right_foot": right_foot
    }

def draw_silhouette(screen, pose, color=WHITE):
    if not pose:
        return
    
    head = pose["head"]
    
    if "left_shoulder" not in pose or "right_shoulder" not in pose:
        return

    left_shoulder = pose["left_shoulder"]
    right_shoulder = pose["right_shoulder"]
    
    neck = ((left_shoulder[0] + right_shoulder[0]) // 2, (left_shoulder[1] + right_shoulder[1]) // 2)

    left_elbow = pose.get("left_elbow", None)
    right_elbow = pose.get("right_elbow", None)
    left_hand = pose.get("left_hand", None)
    right_hand = pose.get("right_hand", None)
    left_hip = pose.get("left_hip", None)
    right_hip = pose.get("right_hip", None)
    left_knee = pose.get("left_knee", None)
    right_knee = pose.get("right_knee", None)
    left_foot = pose.get("left_foot", None)
    right_foot = pose.get("right_foot", None)

    # Draw the head and neck
    pygame.draw.circle(screen, color, head, 65)
    pygame.draw.line(screen, color, head, neck, 20)

    if left_shoulder and right_shoulder:
        pygame.draw.line(screen, color, left_shoulder, right_shoulder, 20)

    # Draw arms
    if left_elbow and left_hand:
        pygame.draw.line(screen, color, left_shoulder, left_elbow, 20)
        pygame.draw.line(screen, color, left_elbow, left_hand, 20)
    if right_elbow and right_hand:
        pygame.draw.line(screen, color, right_shoulder, right_elbow, 20)
        pygame.draw.line(screen, color, right_elbow, right_hand, 20)

    # Draw torso and hips
    if left_hip and right_hip:
        pygame.draw.line(screen, color, left_shoulder, left_hip, 20)
        pygame.draw.line(screen, color, right_shoulder, right_hip, 20)
        pygame.draw.line(screen, color, left_hip, right_hip, 20)
        torso_points = [left_shoulder, right_shoulder, right_hip, left_hip]
        pygame.draw.polygon(screen, color, torso_points)

    # Draw legs
    if left_knee and left_foot:
        pygame.draw.line(screen, color, left_hip, left_knee, 20)
        pygame.draw.line(screen, color, left_knee, left_foot, 20)
    if right_knee and right_foot:
        pygame.draw.line(screen, color, right_hip, right_knee, 20)
        pygame.draw.line(screen, color, right_knee, right_foot, 20)

# Initialize the camera
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
FRAME_HEIGHT, FRAME_WIDTH = frame.shape[:2]

# Generate a random silhouette with limited movements
random_silhouette = generate_limited_silhouette()

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

    # Clear the screen and draw the background
    screen.blit(auto, (0, 0))

    # Draw the player's pose if detected, otherwise draw the random silhouette
    if player_pose:
        draw_silhouette(screen, player_pose, BLACK)
        draw_silhouette(screen, random_silhouette, WHITE)

    # Set game_ready to True after the first frame is displayed
    if not game_ready:
        game_ready = True
        start_time = time.time()

    # Check if the time threshold has been exceeded
    if game_ready:
        elapsed_time = time.time() - start_time
        remaining_time = max(0, time_threshold - elapsed_time)

        # Render the remaining time on the screen
        time_text = font.render(f"Tempo: {int(remaining_time)}", True, RED)
        screen.blit(time_text, (10, 10))

        if elapsed_time > time_threshold:
            # Update only the selected body part in the random silhouette
            if selected_body_part == "Gambe e braccia":
                random_silhouette = generate_limited_silhouette()
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
    pygame.display.flip()
    clock.tick(60)

cap.release()
pygame.quit()