import pygame
import random
import math
import time
import enum
import sys

CELL_SIZE = 25
MAX_SPEED = 2.0

class Actions(enum.Enum):
    ROT_LEFT = 0
    ROT_RIGHT = 1
    ACCELERATE = 2
    DECELERATE = 3


class Game:
    width: int
    height: int
    player_pos: list[float, float, float]
    player_speed: list[float, float]
    flag_pos: list[float, float]
    points: int
    brain: any

    def __init__(self, width: int, height: int, player_pos: list[float, float, float], brain: any = None):
        self.width = width
        self.height = height
        self.player_pos = player_pos[:]
        self.flag_pos = [random.random() * width, random.random() * height]
        self.player_speed = [0.0, 0.0]
        self.brain = brain
        self.points = 0

    def init_draw(self):
        pygame.init()
        self.clock = pygame.time.Clock()
        self.draw_surface = pygame.display.set_mode(((self.width + 2) * CELL_SIZE, (self.height + 2) * CELL_SIZE))
        pygame.display.set_caption("Flag Game")
        self.font = pygame.font.SysFont(None, 25)

    def draw_frame(self):
        # Draw outer wall
        pygame.draw.rect(
            self.draw_surface,
            (255, 0, 0),
            pygame.Rect(0, 0, (self.width + 2) * CELL_SIZE, (self.height + 2) * CELL_SIZE),
        )
        # Draw inner space
        pygame.draw.rect(
            self.draw_surface,
            (255, 200, 200),
            pygame.Rect(CELL_SIZE, CELL_SIZE, self.width * CELL_SIZE, self.height * CELL_SIZE),
        )
        # Draw flag
        pygame.draw.circle(
            self.draw_surface,
            (255, 255, 0),
            (
                (self.flag_pos[0] + 1) * CELL_SIZE,
                (self.flag_pos[1] + 1) * CELL_SIZE,
            ),
            CELL_SIZE,
        )
        # Draw car
        pygame.draw.circle(
            self.draw_surface,
            (255, 0, 255),
            (
                (self.player_pos[0] + 1) * CELL_SIZE,
                (self.player_pos[1] + 1) * CELL_SIZE,
            ),
            CELL_SIZE,
        )
        x = (self.player_pos[0] + 1) * CELL_SIZE
        y = (self.player_pos[1] + 1) * CELL_SIZE
        x_off = math.sin(self.player_pos[2] * math.pi * 2) * CELL_SIZE
        y_off = -math.cos(self.player_pos[2] * math.pi * 2) * CELL_SIZE
        pygame.draw.line(
            self.draw_surface,
            (255, 255, 255),
            (x, y),
            (x + x_off, y + y_off),
        )
        # Draw score
        text = self.font.render(f"Score: {self.points}", True, (255, 255, 255))
        self.draw_surface.blit(text, (5, 5))
        
        pygame.display.flip()

    def get_inputs(self):
        res = set()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)

        if self.brain is None:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_a]:
                res.add(Actions.ROT_LEFT)
            elif keys[pygame.K_d]:
                res.add(Actions.ROT_RIGHT)
            if keys[pygame.K_w]:
                res.add(Actions.ACCELERATE)
            elif keys[pygame.K_s]:
                res.add(Actions.DECELERATE)
            if keys[pygame.K_ESCAPE]:
                sys.exit(0)

        if self.brain is not None:
            raise NotImplemented("I have no brain yet")
            res = brain.get_inputs(self.get_outputs())

        return res


    def tick(self, draw: bool = False):
        inputs = self.get_inputs()  # Get forward/back and/or left/right
        if Actions.ROT_LEFT in inputs:
            self.player_pos[2] = (self.player_pos[2] - 0.02) % 1.0
        elif Actions.ROT_RIGHT in inputs:
            self.player_pos[2] = (self.player_pos[2] + 0.02) % 1.0
        if Actions.ACCELERATE in inputs:
            # Accelerate based on current direction
            x_acc = math.sin(self.player_pos[2] * math.pi * 2) * 0.4
            y_acc = -math.cos(self.player_pos[2] * math.pi * 2) * 0.4
            self.player_speed[0] += x_acc * 0.2
            self.player_speed[1] += y_acc * 0.2
            # Normalize speed so no cheating
            total_speed = math.sqrt((self.player_speed[0] * self.player_speed[0]) + (self.player_speed[1] * self.player_speed[1]))
            if total_speed > MAX_SPEED:
                self.player_speed[0] *= (MAX_SPEED / total_speed)
                self.player_speed[1] *= (MAX_SPEED / total_speed)
        elif Actions.DECELERATE in inputs:
            # Slow down by dampening current momentum
            self.player_speed[0] = 0.0 if abs(self.player_speed[0]) < 0.01 else self.player_speed[0] * 0.8
            self.player_speed[1] = 0.0 if abs(self.player_speed[1]) < 0.01 else self.player_speed[1] * 0.8
        else:
            # Apply lesser dampening
            self.player_speed[0] = 0.0 if abs(self.player_speed[0]) < 0.01 else self.player_speed[0] * 0.98
            self.player_speed[1] = 0.0 if abs(self.player_speed[1]) < 0.01 else self.player_speed[1] * 0.98

        # Move the player!
        new_x = self.player_pos[0] + self.player_speed[0]
        new_y = self.player_pos[1] + self.player_speed[1]
        if not 0.0 < new_x <= self.width:
            if new_x < 0.0:
                new_x = new_x * -0.8
            else:
                new_x = ((new_x - self.width) * -0.8) + self.width
            self.player_speed[0] = self.player_speed[0] * -0.8
        if not 0.0 < new_y <= self.height:
            if new_y < 0.0:
                new_y = new_y * -0.8
            else:
                new_y = ((new_y - self.height) * -0.8) + self.height
            self.player_speed[1] = self.player_speed[1] * -0.8
        self.player_pos[0] = new_x
        self.player_pos[1] = new_y

        # Capture the flag
        distance_to_flag_vec = [self.player_pos[0] - self.flag_pos[0], self.player_pos[1] - self.flag_pos[1]]
        distance_to_flag = math.sqrt((distance_to_flag_vec[0] * distance_to_flag_vec[0]) + (distance_to_flag_vec[1] * distance_to_flag_vec[1]))
        if distance_to_flag < 1.5:
            self.points += 1
            self.flag_pos = [random.random() * self.width, random.random() * self.height]

        if draw:
            game.draw_frame()

        self.clock.tick(60)


game = Game(70, 30, [3, 3, 0.0])
game.init_draw()
while True:
    game.tick(draw=True)
