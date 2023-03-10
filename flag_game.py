import pygame
import random
import math
import time
import enum
import sys

import numpy


CELL_SIZE = 25
MAX_SPEED = 2.0

class Actions(enum.Enum):
    ROT_LEFT = 0
    ROT_RIGHT = 1
    ACCELERATE = 2
    DECELERATE = 3

PLAYERS_PER_GENERATION = 2500
BREEDING_PER_GENERATION = 128


def sigmoid(x):
    """
    Generic sigmoid curve between y=0 and 1, passing through 0,0.5
    """
    return 1 / (1 + math.exp(-x))


def trans_sigmoid(x):
    """
    Sigmoid curve translated to pass through 0,0 and gives a y value
    between -1 and 1
    """
    return 2 * (sigmoid(x) - 0.5)


class Brain:
    inputs = 20
    outputs = 4
    fitness = 0
    num_weights = 80  # 20*4 connections from in to out
    num_biases = 4  # 3 output neurones

    def __init__(self, weights=None, biases=None):
        if not weights:
            weights = [(random.random() * 2) - 1.0 for _ in range(self.num_weights)]
        if not biases:
            biases = [(random.random() * 2) - 1.0 for _ in range(self.num_biases)]
        self.biases = biases
        self.weights = weights
        # TODO: Handle hidden weights here
        # XXX: Hardcoded for now
        self.output_weights = [weights[:20], weights[20:40], weights[40:60], weights[60:80]]
        self.output_bias = biases

    def reset(self):
        self.fitness = 0

    def get_outputs(self, inputs):
        #breakpoint()
        outputs = []
        for o in range(self.outputs):
            output = 0.0
            for i in range(self.inputs):
                output += trans_sigmoid(inputs[i]) * self.output_weights[o][i]
            outputs.append(sigmoid(output + self.output_bias[o]))
        return outputs

    def breed(self, partner, mutation_factor=0.05, mutation_prop=0.2):
        # TODO: Handle different number of neurones/weights between individuals here
        weights = []
        biases = []
        crossover_point = random.randint(0, len(self.weights) + 1)
        for wi in range(len(self.weights)):
            if wi < crossover_point:
                weight = self.weights[wi]
            else:
                weight = partner.weights[wi]
            if random.random() <= mutation_prop:
                weight += ((random.random() * 2.0) - 1.0) * mutation_factor
            if weight > 1.0:
                weight = 1.0
            elif weight < -1.0:
                weight = -1.0
            weights.append(weight)
        for wi in range(len(self.biases)):
            if wi < (crossover_point // 3):
                bias = self.biases[wi]
            else:
                bias = partner.biases[wi]
            if random.random() <= mutation_prop:
                bias += ((random.random() * 2.0) - 1.0) * mutation_factor
            if bias > 1.0:
                bias = 1.0
            elif bias < -1.0:
                bias = -1.0
            biases.append(bias)
        return Brain(weights, biases)

    @staticmethod
    def cross_breed(player_pool, tobreed=BREEDING_PER_GENERATION, total=PLAYERS_PER_GENERATION):
        """
        player_pool should be a list of players to breed sorted with the most
        fit at the beginning of the list.
        """
        breeding_pool = player_pool[:tobreed]
        # Cross breed the top players against each other and apply random mutations
        offspring = []
        for p1 in range(len(breeding_pool)):
            for p2 in range(p1 + 1, len(breeding_pool)):
                offspring.append(breeding_pool[p1].breed(breeding_pool[p2]))
        # Append the parents the to pool then shorten to the required total
        # to fill in any space where new offspring were not created
        offspring.extend(player_pool)
        offspring = offspring[:total]
        for o in offspring:
            o.reset()
        return offspring






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
        self.time = 0

    def draw_frame(self, draw_surface, font, extra_stats):
        # Draw outer wall
        pygame.draw.rect(
            draw_surface,
            (255, 0, 0),
            pygame.Rect(0, 0, (self.width + 2) * CELL_SIZE, (self.height + 2) * CELL_SIZE),
        )
        # Draw inner space
        pygame.draw.rect(
            draw_surface,
            (255, 200, 200),
            pygame.Rect(CELL_SIZE, CELL_SIZE, self.width * CELL_SIZE, self.height * CELL_SIZE),
        )
        # Draw flag
        pygame.draw.circle(
            draw_surface,
            (255, 255, 0),
            (
                (self.flag_pos[0] + 1) * CELL_SIZE,
                (self.flag_pos[1] + 1) * CELL_SIZE,
            ),
            CELL_SIZE,
        )
        # Draw car
        pygame.draw.circle(
            draw_surface,
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
            draw_surface,
            (255, 255, 255),
            (x, y),
            (x + x_off, y + y_off),
            width = 3,
        )
        # Draw collisions
        for angle in range(16):
            intensity, x_off, y_off = self.find_collision(angle / 16)
            pygame.draw.line(
                draw_surface,
                (0, int(255 * intensity), int(255 * intensity)),
                (x, y),
                (x + (x_off * CELL_SIZE), y + (y_off * CELL_SIZE)),
            )
        # Draw score
        text_edge = ((self.width + 2) * CELL_SIZE) + 5
        pygame.draw.rect(
            draw_surface,
            (50, 50, 50),
            pygame.Rect(text_edge, 0, text_edge + 300, (self.height + 2) * CELL_SIZE),
        )

        text = font.render(f"Time: {self.time}", True, (255, 255, 255))
        draw_surface.blit(text, (text_edge, 5))
        text = font.render(f"Score: {self.points}", True, (255, 255, 255))
        draw_surface.blit(text, (text_edge, 30))

        res = self.get_outputs()
        text = font.render(f"I(SP_TOT): {res[0]}", True, (255, 255, 255))
        draw_surface.blit(text, (text_edge, 55))
        text = font.render(f"I(SP_ANG): {res[1]}", True, (255, 255, 255))
        draw_surface.blit(text, (text_edge, 80))
        text = font.render(f"I(FL_CLO): {res[2]}", True, (255, 255, 255))
        draw_surface.blit(text, (text_edge, 105))
        text = font.render(f"I(FL_ANG): {res[3]}", True, (255, 255, 255))
        draw_surface.blit(text, (text_edge, 130))
        for angle in range(16):
            text = font.render(f"I(COL{angle:02}): {res[4 + angle]}", True, (255, 255, 255))
            draw_surface.blit(text, (text_edge, 155 + (25 * angle)))

        for i, stat_key in enumerate(extra_stats):
            text = font.render(f"{stat_key}: {extra_stats[stat_key]}", True, (255, 255, 255))
            draw_surface.blit(text, (text_edge, 580 + (25 * i)))


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
            outputs = self.brain.get_outputs(self.get_outputs())
            if outputs[0] > outputs[1] and outputs[0] > 0.7:
                res.add(Actions.ROT_LEFT)
            elif outputs[1] > outputs[0] and outputs[1] > 0.7:
                res.add(Actions.ROT_RIGHT)
            if outputs[2] > outputs[3] and outputs[2] > 0.7:
                res.add(Actions.ACCELERATE)
            elif outputs[3] > outputs[2] and outputs[3] > 0.7:
                res.add(Actions.DECELERATE)

        return res


    def find_collision(self, angle, length=30):
        # Find closeness from self at a rotation of angle (0.0 to 1.0) to the edge
        # 1. Get current location and extend forward by length
        x = self.player_pos[0]
        y = self.player_pos[1]
        x_off = math.sin((self.player_pos[2] + angle) * math.pi * 2) * length
        y_off = -math.cos((self.player_pos[2] + angle) * math.pi * 2) * length

        if x + x_off < 0:
            m = abs(x / x_off)
            x_off *= m
            y_off *= m
        elif x + x_off > self.width:
            # Intersection multiple is (self.width - x) / ((self.width - x) + x_off)
            m = abs((self.width - x) / x_off)
            x_off *= m
            y_off *= m

        if y + y_off < 0:
            m = abs(y / y_off)
            x_off *= m
            y_off *= m
        elif y + y_off > self.height:
            # Intersection multiple is (self.width - x) / ((self.width - x) + x_off)
            m = abs((self.height - y) / y_off)
            x_off *= m
            y_off *= m

        distance = math.sqrt((x_off * x_off) + (y_off * y_off))
        if distance > length:
            return 0.0, x_off, y_off
        return min(((30 - distance) / length, 1.0)), x_off, y_off


    def get_outputs(self):
        distance_to_flag_vec = [self.player_pos[0] - self.flag_pos[0], self.player_pos[1] - self.flag_pos[1]]
        closeness_to_flag = min(1.0, 2 / math.sqrt((distance_to_flag_vec[0] * distance_to_flag_vec[0]) + (distance_to_flag_vec[1] * distance_to_flag_vec[1])))

        flag_bearing = ((math.atan2(distance_to_flag_vec[1], distance_to_flag_vec[0]) / (2 * math.pi)) + 0.25) % 1.0
        flag_relative_bearing = (((flag_bearing - self.player_pos[2]) % 1) * 2.0) - 1.0

        speed_total = math.sqrt((self.player_speed[0] * self.player_speed[0]) + (self.player_speed[1] * self.player_speed[1])) / MAX_SPEED
        if speed_total > 0:
            speed_bearing = ((math.atan2(self.player_speed[1], self.player_speed[0]) / (2 * math.pi)) + 0.25) % 1.0
            speed_relative_bearing = ((((self.player_pos[2] + 0.5) - speed_bearing) % 1) * 2.0) - 1.0
        else:
            speed_relative_bearing = 0.0

        res = [
            speed_total,                       # Current speed amplitude
            speed_relative_bearing,            # Speed vector angle to facing
            closeness_to_flag,                 # Higher if we are closer to the flag (inverse distance to flag)
            flag_relative_bearing,             # Rotation relative to us towards flag (0 == ahead, 0.5 == right 90deg, -0.5 == left 90deg)
        ]
        for angle in range(16):
            intensity, _, _ = self.find_collision(angle / 16)
            res.append(intensity)

        return res


    def tick(self):
        self.time += 1
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

        ## All point related things
        # Capture the flag + 1000
        distance_to_flag_vec = [self.player_pos[0] - self.flag_pos[0], self.player_pos[1] - self.flag_pos[1]]
        distance_to_flag = math.sqrt((distance_to_flag_vec[0] * distance_to_flag_vec[0]) + (distance_to_flag_vec[1] * distance_to_flag_vec[1]))
        if distance_to_flag < 1.5:
            self.points += 1000
            self.flag_pos = [random.random() * self.width, random.random() * self.height]

        # Facing flag + 5
        flag_bearing = ((math.atan2(distance_to_flag_vec[1], distance_to_flag_vec[0]) / (2 * math.pi)) + 0.25) % 1.0
        flag_relative_bearing = (((flag_bearing - self.player_pos[2]) % 1) * 2.0) - 1.0
        if -0.1 < flag_relative_bearing < 0.1:
            self.points += 5

        # Moving
        if abs(self.player_speed[0]) + abs(self.player_speed[1]) > 0.5:
            self.points += 2

        # Idle, punish!
        if not inputs or (not self.player_speed[0] and not self.player_speed[1]):
            self.points -= 2

WIDTH = 70
HEIGHT = 30

pygame.init()
clock = pygame.time.Clock()
draw_surface = pygame.display.set_mode((((WIDTH + 2) * CELL_SIZE) + 300, (HEIGHT + 2) * CELL_SIZE))
pygame.display.set_caption("Flag Game")
font = pygame.font.SysFont(None, 25)


game = Game(WIDTH, HEIGHT, [3, 3, 0.0], brain=Brain())


player_pool = [Brain() for _ in range(PLAYERS_PER_GENERATION)]
all_games = [Game(WIDTH, HEIGHT, [3, 3, 0.0], brain=p) for p in player_pool]
death_pool = []

ROUND_LENGTH = 1000

ticks = 0
generation = 1
following = 1
stats = {
    "Following": following,
    "#Players": PLAYERS_PER_GENERATION,
    "Generation": generation,
    "Av. Fitness": 0,
}
while True:
    # Collect dead players with negative scores < -50
    ticks += 1
    players = 0
    best_player = [0, 0]
    for i, game in enumerate(all_games):
        if game.points > best_player[1]:
            best_player = [i, game.points]
        if i == following:
            game.draw_frame(draw_surface, font, stats)
        if game.points > -50:
            players += 1
            game.tick()
    if ticks % 100 == 0:  # Stop shifting around too quickly!
        following = best_player[0]
    stats["Following"] = following
    stats["#Players"] = players
    clock.tick(100)
    if players and ticks < ROUND_LENGTH:
        continue

    # End of generation!
    death_pool = sorted(all_games, key=lambda x: x.points, reverse=True)
    fitness_stats = [x.points for x in death_pool]
    ps = numpy.percentile(fitness_stats, q=(99, 95, 90, 50, 25))
    top = max(fitness_stats)
    avg = numpy.mean(fitness_stats)
    std = numpy.std(fitness_stats)
    print(f"End of generation {generation}. Fitness top:{top:0.4f} mean:{avg:0.4f} std:{std:0.4f}, 99/95/90/50/25 percentiles:", ps)

    # Iterate the generation!
    generation += 1
    stats["Generation"] = generation
    dead_players = [x.brain for x in death_pool]
    player_pool = Brain.cross_breed(dead_players)
    all_games = [Game(WIDTH, HEIGHT, [3, 3, 0.0], brain=p) for p in player_pool]
    ticks = 0
