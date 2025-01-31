import os
import pygame
import sys
import cv2
import mediapipe as mp
import random
import numpy as np
from pygame.locals import *
import time  # Importa il modulo time per gestire il tempo

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
start_time = time.time()  # Record the start time

# Font for displaying the remaining time
font = pygame.font.Font(None, 74)  # Usa un font predefinito con dimensione 74

# Variable to check if the game is ready
game_ready = False

def generate_limited_silhouette():
    # Generate a silhouette with limited movements
    center_x, center_y = WIDTH // 2, HEIGHT // 2
    offset_range = 20  # Reduced range of random offset for limited movements

    # Randomize the positions of the joints with limited range
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
    
    # Extract key points
    head = pose["head"]
    
    # Check if key points exist
    if "left_shoulder" not in pose or "right_shoulder" not in pose:
        return  # Exit if shoulders are not found

    left_shoulder = pose["left_shoulder"]
    right_shoulder = pose["right_shoulder"]
    
    # Calculate neck position as the midpoint between the shoulders
    neck = ((left_shoulder[0] + right_shoulder[0]) // 2, (left_shoulder[1] + right_shoulder[1]) // 2)

    # Check if other points are available
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

    # Draw the head and a line connecting it to the neck
    pygame.draw.circle(screen, color, head, 65)  # Larger head
    pygame.draw.line(screen, color, head, neck,  20)  # Line from head to neck

    if left_shoulder and right_shoulder:
        pygame.draw.line(screen, color, left_shoulder, right_shoulder, 20)

    # Draw lines for arms and hands only if points are available
    if left_elbow and left_hand:
        pygame.draw.line(screen, color, left_shoulder, left_elbow, 20)
        pygame.draw.line(screen, color, left_elbow, left_hand, 20)

    if right_elbow and right_hand:
        pygame.draw.line(screen, color, right_shoulder, right_elbow, 20)
        pygame.draw.line(screen, color, right_elbow, right_hand, 20)

    # Draw lines for the torso and hips
    if left_hip and right_hip:
        pygame.draw.line(screen, color, left_shoulder, left_hip, 20)
        pygame.draw.line(screen, color, right_shoulder, right_hip, 20)
        pygame.draw.line(screen, color, left_hip, right_hip, 20)

        # Fill the torso with a polygon
        torso_points = [
            left_shoulder,
            right_shoulder,
            right_hip,
            left_hip
        ]
        pygame.draw.polygon(screen, color, torso_points)  # Fill the torso

    # Draw lines for the legs and feet
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

game_running = True
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
    screen.blit(auto, (0, 0))  # Draw the background first

    # Draw the player's pose if detected, otherwise draw the random silhouette
    if player_pose:
        draw_silhouette(screen, player_pose, BLACK)  # Draw the player's silhouette in black
        draw_silhouette(screen, random_silhouette, WHITE)  # Draw the random silhouette in white

    # Set game_ready to True after the first frame is displayed
    if not game_ready:
        game_ready = True
        start_time = time.time()  # Start the timer only after the game is ready

    # Check if the time threshold has been exceeded
    if game_ready:
        elapsed_time = time.time() - start_time
        remaining_time = max(0, time_threshold - elapsed_time)  # Calcola il tempo rimanente

        # Render the remaining time on the screen
        time_text = font.render(f"Tempo: {int(remaining_time)}", True, RED)
        screen.blit(time_text, (10, 10))  # Posiziona il testo in alto a sinistra

        if elapsed_time > time_threshold:
            random_silhouette = generate_limited_silhouette()  # Generate a new silhouette
            start_time = time.time()  # Reset the start time
            time_threshold = max(10, time_threshold - 0.5)  # Decrease the time threshold, but not below 10 seconds

    # Update the display
    pygame.display.flip()
    clock.tick(60)

cap.release()
pygame.quit()