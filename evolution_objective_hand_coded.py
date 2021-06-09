import random, sys, time, os, json, re
import numpy as np
from collections import Counter, deque

class hand_coded_model:
    def __init__(self, index):
        self.index = index
        return

    def get_action(self, state, direction, objective):
        # Figure out where the objective is in relation to the agent
        agent_index = 50+self.index
        agent_objective = objective
        agent_pos = np.argwhere(state==agent_index)
        agent_ypos = agent_pos[0][0]
        agent_xpos = agent_pos[0][1]
        objective_pos = np.argwhere(state==agent_objective)
        objective_ypos = objective_pos[0][0]
        objective_xpos = objective_pos[0][1]
        objective_x_direction = 0
        if objective_xpos > agent_xpos:
            objective_x_direction = 1
        elif objective_xpos < agent_xpos:
            objective_x_direction = -1
        objective_y_direction = 0
        if objective_ypos > agent_ypos:
            objective_y_direction = 1
        elif objective_ypos < agent_ypos:
            objective_y_direction = -1
        # Based on which direction the agent is facing
        # and whether there is free space ahead
        # decide whether go forward or turn
        tile_ahead = 0
        if direction == 0: # up
            tile_ahead = state[agent_ypos-1][agent_xpos]
        elif direction == 1: # right
            tile_ahead = state[agent_ypos][agent_xpos+1]
        elif direction == 2: # down
            tile_ahead = state[agent_ypos+1][agent_xpos]
        elif direction == 3: # right
            tile_ahead = state[agent_ypos][agent_xpos-1]
        if tile_ahead == objective:
            return 1
        if tile_ahead == 0:
            if random.choice([0,1]) == 1:
                return 1

        continue_forward = False
        if direction == 0: # up
            if objective_y_direction < 0:
                continue_forward = True
        elif direction == 1: # right
            if objective_x_direction > 0:
                continue_forward = True
        elif direction == 2: # down
            if objective_y_direction > 0:
                continue_forward = True
        elif direction == 3: # right
            if objective_x_direction < 0:
                continue_forward = True
        if continue_forward == True:
            if tile_ahead != 0:
                continue_forward = False
        if continue_forward == True:
            return 1 # move forward
        else:
            return random.choice([2,3])

class agent:
    def __init__(self, index, x, y, objective):
        self.index = index
        self.xpos = x
        self.ypos = y
        self.objective = objective
        self.direction = random.choice([0,1,2,3]) # 0=up, 1=right, 2=down, 3=left
        self.model = hand_coded_model(index)
        self.state = None

    def respawn(self, x, y, objective):
        self.xpos = x
        self.ypos = y
        self.objective = objective
        self.direction = random.choice([0,1,2,3]) # 0=up, 1=down, 2=left, 3=right

    def get_action(self):
        return self.model.get_action(self.state, self.direction, self.objective)


class game_space:
    def __init__(self, width, height, num_walls=10, num_agents=1, num_objectives=2):
        self.num_agents = num_agents
        self.width = width
        self.height = height
        self.num_actions = 3
        self.start_objectives = num_objectives
        self.start_walls = num_walls
        self.reset()

    def reset(self):
        self.agents = []
        self.walls = []
        self.objectives = []
        self.num_walls = len(self.walls)
        space = self.make_empty_game_space()
        self.game_space = space
        self.add_walls(self.start_walls)
        self.initial_game_space = np.array(self.game_space)
        self.create_objectives(self.start_objectives)
        self.add_objectives()
        for index in range(self.num_agents):
            self.create_new_agent()
        state = self.get_state()
        for index in range(self.num_agents):
            self.agents[index].state = state

    def step(self):
        for index in range(len(self.agents)):
            action = gs.agents[index].get_action()
            self.move_agent(index, action)

    def make_empty_game_space(self):
        space = np.zeros((self.height, self.width), dtype=int)
        for n in range(self.width):
            space[0][n] = 1
            space[self.height-1][n] = 1
        for n in range(self.height):
            space[n][0] = 1
            space[n][self.width-1] = 1
        return space

    def add_blocks(self, num):
        locations = self.get_random_empty_space(num)
        for item in locations:
            y, x = item
            self.game_space[y][x] = 1
            self.walls.append([y, x])
        self.num_walls = len(self.walls)

    def add_walls(self, num):
        added = 0
        sequence_max = 10
        height, width = self.game_space.shape
        while added < num:
            item = self.get_random_empty_space(1)
            y, x  = item[0]
            self.game_space[y][x] = 1
            self.walls.append([y, x])
            s_len = 0
            while s_len < sequence_max:
                move = random.randint(0,3)
                if move == 0:
                    x = max(1, x-1)
                elif move == 1:
                    x = min(width-2, x+1)
                elif move == 2:
                    y = max(1, y-1)
                elif move == 3:
                    y = min(height-2, y+1)
                if self.game_space[y][x] == 0:
                    added += 1
                    s_len += 1
                    self.game_space[y][x] = 1
                    self.walls.append([y, x])
                if added >= num:
                    break

    def add_objectives(self):
        for index, item in enumerate(self.objectives):
            y, x = item
            self.game_space[y][x] = 1000 + index

    def create_objectives(self, num):
        num_objectives = len(self.objectives)
        locations = self.get_random_empty_space(num)
        for item in locations:
            self.objectives.append(item)
        self.num_objectives = len(self.objectives)

    def move_objective(self, index):
        old = self.objectives[index]
        oldy, oldx = old
        item = self.get_random_empty_space(1)
        newy, newx = item[0]
        self.game_space[newy][newx] = 1000 + index
        if self.game_space[oldy][oldx] == 1000 + index:
            self.game_space[oldy][oldx] = 0
        self.objectives[index] = item[0]

    def create_new_agent(self):
        index = len(self.agents)
        item = self.get_random_empty_space(1)
        ypos, xpos = item[0]
        objective = random.choice(range(len(self.objectives))) + 1000
        new_agent = agent(index, xpos, ypos, objective)
        self.agents.append(new_agent)
        state = self.get_state()
        self.agents[index].state = state

    def respawn_agent(self, index):
        item = self.get_random_empty_space(1)
        ypos, xpos = item[0]
        objective = random.choice(range(len(self.objectives))) + 1000
        self.agents[index].respawn(xpos, ypos, objective)
        state = self.get_state()
        self.agents[index].state = state

    def add_items_to_game_space(self):
        space = np.array(self.game_space)
        if len(self.agents) > 0:
            for agent in self.agents:
                space[agent.ypos][agent.xpos] = 50+agent.index
        return space

    def get_random_empty_space(self, num=1):
        space = self.add_items_to_game_space()
        empties = list(np.argwhere(space == 0))
        return random.sample(empties, num)

    def move_forward(self, index):
        direction = self.agents[index].direction
        xpos = self.agents[index].xpos
        ypos = self.agents[index].ypos
        newx = xpos
        newy = ypos
        if direction == 0: # moving up
            newy = ypos-1
        elif direction == 1: # moving right
            newx = xpos+1
        elif direction == 2: # moving down
            newy = ypos+1
        elif direction == 3: # moving left
            newx = xpos-1
        space = self.add_items_to_game_space()
        space_val = space[newy][newx]
        if space_val == 0: # empty space, move forward
            self.agents[index].xpos = newx
            self.agents[index].ypos = newy
        return space_val

    def move_agent(self, index, action):
        done = False
        direction = self.agents[index].direction
        if action == 1: # move forward
            space_val = self.move_forward(index)
            if space_val >= 1000: #bumping into an objective ends the episode
                done = True
                agent_objective = self.agents[index].objective
                if space_val == agent_objective:
                    self.move_objective(space_val-1000)
            elif space_val >= 50 and space_val < 100:
                done = True # bumping into another agent ends the episode
        elif action == 2: # rotate clockwise
            direction += 1
            if direction > 3:
                direction = 0
        elif action == 3: # rotate anticlockwise
            direction -= 1
            if direction < 0:
                direction = 3
        state = self.get_state()
        self.agents[index].state = state
        self.agents[index].direction = direction
        if done == True:
            self.respawn_agent(index)

    def get_state(self):
        space = self.add_items_to_game_space()
        state = np.array(space)
        return state

    def get_printable(self, item):
        if item == 1:
            return "\x1b[1;37;40m" + "░" + "\x1b[0m"
        elif item >= 50 and item < 100:
            agent_index = item - 50
            agent = self.agents[agent_index]
            objective = self.agents[agent_index].objective
            d = agent.direction
            c = 30 + (1000-objective)%7
            if d == 0:
                return "\x1b[1;" + str(c) + ";40m" + "^" + "\x1b[0m"
            elif d == 1:
                return "\x1b[1;" + str(c) + ";40m" + ">" + "\x1b[0m"
            elif d == 2:
                return "\x1b[1;" + str(c) + ";40m" + "v" + "\x1b[0m"
            elif d == 3:
                return "\x1b[1;" + str(c) + ";40m" + "<" + "\x1b[0m"
        elif item >= 1000:
            label = "X"
            c = 30 + (1000-item)%7
            return "\x1b[0;" + str(c) + ";40m" + str(label) + "\x1b[0m"
        else:
            return "\x1b[1;32;40m" + " " + "\x1b[0m"

    def print_game_space(self):
        printable = ""
        space = self.add_items_to_game_space()
        for column in space:
            for item in column:
                printable += self.get_printable(item)
            printable += "\n"
        return printable


# Train the game
game_space_width = 60
game_space_height = 30
num_walls = 150
num_agents = 70
num_objectives = 7

gs = game_space(game_space_width,
                game_space_height,
                num_walls,
                num_agents=num_agents,
                num_objectives=num_objectives)

print_visuals = True
steps = 0
while True:
    gs.step()
    os.system('clear')
    print()
    print(gs.print_game_space())
    print()
    time.sleep(0.03)
    steps += 1

