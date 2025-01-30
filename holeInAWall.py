import os
import pygame
import sys
import cv2
import mediapipe as mp
import random
import numpy as np
from pygame.locals import *

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

def generate_random_silhouette():
    # Generate a random silhouette within the center of the screen
    center_x, center_y = WIDTH // 2, HEIGHT // 2
    offset_range = 50  # Range of random offset from the center

    # Randomize the positions of the joints
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

def draw_silhouette(screen, pose):
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
    pygame.draw.circle(screen, BLACK, head, 65)  # Larger head
    pygame.draw.line(screen, BLACK, head, neck,  20)  # Line from head to neck

    if left_shoulder and right_shoulder:
        pygame.draw.line(screen, BLACK, left_shoulder, right_shoulder, 20)

    # Draw lines for arms and hands only if points are available
    if left_elbow and left_hand:
        pygame.draw.line(screen, BLACK, left_shoulder, left_elbow, 20)
        pygame.draw.line(screen, BLACK, left_elbow, left_hand, 20)

    if right_elbow and right_hand:
        pygame.draw.line(screen, BLACK, right_shoulder, right_elbow, 20)
        pygame.draw.line(screen, BLACK, right_elbow, right_hand, 20)

    # Draw lines for the torso and hips
    if left_hip and right_hip:
        pygame.draw.line(screen, BLACK, left_shoulder, left_hip, 20)
        pygame.draw.line(screen, BLACK, right_shoulder, right_hip, 20)
        pygame.draw.line(screen, BLACK, left_hip, right_hip, 20)

    # Draw lines for the legs and feet
    if left_knee and left_foot:
        pygame.draw.line(screen, BLACK, left_hip, left_knee, 20)
        pygame.draw.line(screen, BLACK, left_knee, left_foot, 20)

    if right_knee and right_foot:
        pygame.draw.line(screen, BLACK, right_hip, right_knee, 20)
        pygame.draw.line(screen, BLACK, right_knee, right_foot, 20)



# Initialize the camera
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
FRAME_HEIGHT, FRAME_WIDTH = frame.shape[:2]

# Generate a random silhouette
random_silhouette = generate_random_silhouette()

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
    screen.blit(auto, (0, 0))

    # Draw the player's pose if detected, otherwise draw the random silhouette
    if player_pose:
        draw_silhouette(screen, player_pose)
    else:
        draw_silhouette(screen, random_silhouette)

    # Update the display
    pygame.display.flip()
    clock.tick(30)

cap.release()
pygame.quit()