import pygame
import random

# Inizializza Pygame
pygame.init()

# Dimensioni della finestra
WIDTH, HEIGHT = 800, 600

# Colori
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

# Inizializza la finestra
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Hole in the Wall")

# Clock per il framerate
clock = pygame.time.Clock()

# Classe per il giocatore
class Player:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.size = 50

    def move(self, dx, dy):
        self.x += dx
        self.y += dy
        
        # Limita il movimento ai bordi della finestra
        self.x = max(0, min(WIDTH, self.x))
        self.y = max(0, min(HEIGHT, self.y))

    def draw(self, screen):
        pygame.draw.ellipse(screen, GREEN, (self.x - self.size // 2, self.y - self.size // 2, self.size, self.size))

    def get_bounds(self):
        return pygame.Rect(self.x - self.size // 2, self.y - self.size // 2, self.size, self.size)

# Classe per il muro
class Wall:
    def __init__(self):
        self.x = WIDTH
        self.width = 50
        self.gap_height = 120
        self.gap_y = random.randint(0, HEIGHT - self.gap_height)
        self.speed = 5

    def update(self):
        self.x -= self.speed
        # Resetta il muro se esce dallo schermo
        if self.x + self.width < 0:
            self.x = WIDTH
            self.gap_y = random.randint(0, HEIGHT - self.gap_height)

    def draw(self, screen):
        # Parte superiore del muro
        pygame.draw.rect(screen, WHITE, (self.x, 0, self.width, self.gap_y))
        # Parte inferiore del muro
        pygame.draw.rect(screen, WHITE, (self.x, self.gap_y + self.gap_height, self.width, HEIGHT - (self.gap_y + self.gap_height)))

    def collides(self, player):
        player_rect = player.get_bounds()
        # Controlla collisione con la parte superiore o inferiore del muro
        if player_rect.colliderect(pygame.Rect(self.x, 0, self.width, self.gap_y)) or \
           player_rect.colliderect(pygame.Rect(self.x, self.gap_y + self.gap_height, self.width, HEIGHT - (self.gap_y + self.gap_height))):
            return True
        return False

# Inizializza il giocatore e il muro
player = Player(WIDTH // 2, HEIGHT - 100)
wall = Wall()

# Stato del gioco
game_running = True

# Ciclo principale del gioco
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

    if game_running:
        # Controlla input da tastiera
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            player.move(-5, 0)
        if keys[pygame.K_RIGHT]:
            player.move(5, 0)
        if keys[pygame.K_UP]:
            player.move(0, -5)
        if keys[pygame.K_DOWN]:
            player.move(0, 5)

        # Aggiorna il muro
        wall.update()

        # Controlla collisioni
        if wall.collides(player):
            game_running = False

    # Disegna lo schermo
    screen.fill(BLACK)

    if game_running:
        player.draw(screen)
        wall.draw(screen)
    else:
        # Schermata di Game Over
        font = pygame.font.Font(None, 74)
        text = font.render("Game Over!", True, RED)
        screen.blit(text, (WIDTH // 2 - text.get_width() // 2, HEIGHT // 2 - text.get_height() // 2))

    pygame.display.flip()
    clock.tick(60)
