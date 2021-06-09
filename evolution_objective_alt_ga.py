import random, sys, time, os, json, re, pickle
import numpy as np
from collections import Counter, deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical

class Net(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, fc1_weights, fc2_weights):
        super(Net, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(self.state_size, self.hidden_size, bias=False)
        self.fc2 = nn.Linear(self.hidden_size, self.action_size, bias=False)
        self.fc1.weight.data =  fc1_weights
        self.fc2.weight.data =  fc2_weights

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(F.relu(self.fc2(x)))
        return x

    def get_action(self, state):
        with torch.no_grad():
            state = state.float()
            ret = self.forward(Variable(state))
            action = torch.argmax(ret)
            return action

class GN_model:
    def __init__(self, state_size, action_size, hidden_size, fc1_weights, fc2_weights):
        self.policy = Net(state_size, action_size, hidden_size, fc1_weights, fc2_weights)

    def get_action(self, state):
        action = self.policy.get_action(state)
        return action


class agent:
    def __init__(self, x, y, objective, state_size, action_size, hidden_size, genome, gi):
        self.xpos = x
        self.ypos = y
        self.objective = objective
        self.gi = gi
        fc1_weights = torch.Tensor(np.reshape(genome[:state_size*hidden_size],
                                              (hidden_size, state_size)))
        fc2_weights = torch.Tensor(np.reshape(genome[state_size*hidden_size:],
                                              (action_size, hidden_size)))
        self.model = GN_model(state_size, action_size, hidden_size, fc1_weights, fc2_weights)
        self.state = None
        self.episode_steps = 0

    def get_action(self):
        return self.model.get_action(self.state)

class game_space:
    def __init__(self, width, height, num_walls=0,
                 num_agents=1, num_objectives=1,
                 max_episode_len=200,
                 savedir="save_gn"):
        self.max_episode_len = max_episode_len
        self.num_agents = num_agents
        self.savedir = savedir
        if not os.path.exists(self.savedir):
            os.makedirs(self.savedir)
        self.width = width
        self.height = height
        self.start_objectives = num_objectives
        self.start_walls = num_walls
        self.obj_start = 2000
        self.ag_start = 2
        self.action_size = 5
        self.hidden_size = 16
        self.pool_size = 5000
        self.state_size = self.get_state_size()
        self.genome_size = (self.state_size*self.hidden_size) + (self.action_size*self.hidden_size)
        self.genome_pool = []
        if os.path.exists("genome_pool.pkl"):
            print("Loading genomes.")
            self.genome_pool = self.load_genomes()
        else:
            print("Creating genomes.")
            self.genome_pool = self.make_genomes()
        self.reset()

    def reset(self):
        self.agents = []
        for index in range(self.num_agents):
            self.agents.append(None)
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
            self.create_new_agent(index)
        for index, agent in enumerate(self.agents):
            state = self.get_agent_state(index)
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

    def add_walls(self, num):
        locations = self.get_random_empty_space(num)
        for item in locations:
            y, x = item
            self.game_space[y][x] = 1
            self.walls.append([y, x])
        self.num_walls = len(self.walls)

    def add_objectives(self):
        for index, item in enumerate(self.objectives):
            y, x = item
            self.game_space[y][x] = self.obj_start + (index*10)

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
        self.game_space[newy][newx] = self.obj_start + (index*10)
        if self.game_space[oldy][oldx] == self.obj_start + (index*10):
            self.game_space[oldy][oldx] = 0
        self.objectives[index] = item[0]

    def create_new_agent(self, index):
        item = self.get_random_empty_space(1)
        ypos, xpos = item[0]
        objective = self.obj_start + (random.choice(range(len(self.objectives)))*10)
        state_size = self.get_state_size()
        action_size = self.action_size
        found = False
        gi = 0
        genome = None
        reward = None
        while found == False:
            gi = random.choice(range(len(self.genome_pool)))
            item = self.genome_pool[gi]
            genome, reward = item
            if reward is None:
                found = True

        new_agent = agent(xpos, ypos, objective, state_size,
                          action_size, self.hidden_size, genome, gi)
        self.agents[index] = new_agent

    def add_items_to_game_space(self):
        space = np.array(self.game_space)
        if len(self.agents) > 0:
            for index, agent in enumerate(self.agents):
                if agent is not None:
                    space[agent.ypos][agent.xpos] = self.ag_start+index
        return space

    def get_random_empty_space(self, num=1):
        space = self.add_items_to_game_space()
        empties = list(np.argwhere(space == 0))
        return random.sample(empties, num)

    def move_forward(self, index, action):
        xpos = self.agents[index].xpos
        ypos = self.agents[index].ypos
        newx = xpos
        newy = ypos
        if action == 1: # moving up
            newy = ypos-1
        elif action == 2: # moving right
            newx = xpos+1
        elif action == 3: # moving down
            newy = ypos+1
        elif action == 4: # moving left
            newx = xpos-1
        space = self.add_items_to_game_space()
        space_val = space[newy][newx]
        if space_val == 0: # empty space, move forward
            self.agents[index].xpos = newx
            self.agents[index].ypos = newy
        return space_val

    def move_agent(self, index, action):
        done = False
        reward = 0
        if action == 0: # do nothing
            pass
        else:
            space_val = self.move_forward(index, action)
            if space_val >= self.obj_start: #bumping into an objective ends the episode
                done = True
                agent_objective = self.agents[index].objective
                if space_val == agent_objective:
                    ml = gs.max_episode_len
                    es = self.agents[index].episode_steps
                    #reward = min(0.5, ((ml - es)/ml)) # higher reward for doing it faster
                    reward = 1 # only rewards are positive, for achieving the objective
                    self.move_objective(int((space_val-self.obj_start)/10))
                else:
                    reward = -1
                    done = True # bumping into the wrong objective ends the episode
            elif space_val >= self.ag_start and space_val < self.obj_start:
                reward = -1
                done = True # bumping into another agent ends the episode
            self.agents[index].direction = action
        state = self.get_agent_state(index)
        self.agents[index].state = state
        self.agents[index].episode_steps += 1

        if self.agents[index].episode_steps >= self.max_episode_len:
            done = True
        if done == True:
            gi = self.agents[index].gi
            item = self.genome_pool[gi]
            genome, fitness = item
            fitness = reward
            self.genome_pool[gi] = [genome, fitness]
            self.create_new_agent(index)
            state = self.get_agent_state(index)
            self.agents[index].state = state

    def get_agent_state(self, index):
        return self.make_small_state(index)
        space = self.add_items_to_game_space()
        objective = self.agents[index].objective
        state = [objective] + list(np.ravel(space))
        state = np.array(state)
        state = torch.FloatTensor(state)
        state = state.unsqueeze(0)
        return state

    def get_state_size(self):
        #state_size = (self.height*self.width) + 1
        state_size = 10
        return state_size

    def get_tile_val(self, tile, objective):
        if tile == 0:
            return 0
        if tile == objective:
            return 1
        return -1

    def make_small_state(self, index):
        space = self.add_items_to_game_space()
        objective = self.agents[index].objective
        agent_index = self.ag_start+index
        agent_objective = objective
        agent_pos = np.argwhere(space==agent_index)
        agent_ypos = agent_pos[0][0]
        agent_xpos = agent_pos[0][1]
        objective_pos = np.argwhere(space==agent_objective)
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

        tiles = []
        offsets = [[-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1]]
        for i in offsets:
            oy, ox = i
            tile = self.get_tile_val(space[agent_ypos+oy][agent_xpos+ox], objective)
            tiles.append(tile)
        state = [objective_x_direction, objective_y_direction]
        state.extend(tiles)
        state = np.array(state)
        state = torch.FloatTensor(state)
        state = state.unsqueeze(0)
        return state



    def get_printable(self, item):
        if item == 1:
            return "\x1b[1;37;40m" + "░" + "\x1b[0m"
        elif item >= self.ag_start and item < self.obj_start:
            return "\x1b[1;32;40m" + "x" + "\x1b[0m"
        elif item >= self.obj_start:
            return "\x1b[1;35;40m" + "o" + "\x1b[0m"
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

    def load_stats(self):
        self.stats_save_path = os.path.join(self.savedir, "stats.json")
        if os.path.exists(self.stats_save_path):
            with open(self.stats_save_path, "r") as f:
                self.stats = json.loads(f.read())

    def save_stats(self):
        self.stats_save_path = os.path.join(self.savedir, "stats.json")
        with open(self.stats_save_path, "w") as f:
            f.write(json.dumps(self.stats))

    def make_genomes(self):
        genome_pool = []
        for _ in range(self.pool_size):
            genome = np.random.uniform(-1, 1, self.genome_size)
            fitness = None
            genome_pool.append([genome, fitness])
        return genome_pool

    def reproduce_genome(self, g1, g2, num_offspring):
        new_genomes = []
        for _ in range(num_offspring):
            s = random.randint(10, len(g1)-10)
            g3 = np.concatenate((g1[:s], g2[s:]))
            new_genomes.append(g3)
            g4 = np.concatenate((g2[:s], g1[s:]))
            new_genomes.append(g4)
        return new_genomes

    def mutate_genome(self, g, num_mutations):
        new_genomes = []
        for _ in range(num_mutations):
            n = int(0.001 * len(g))
            indices = random.sample(range(len(g)), n)
            gm = g
            for index in indices:
                val = random.uniform(-1, 1)
                gm[index] = val
            new_genomes.append(gm)
        return new_genomes

    def create_new_genome_pool(self):
        # Get genomes from previous pool that had positive fitness
        fit_genomes = []
        for item in self.genome_pool:
            genome, fitness = item
            if fitness is not None:
                if fitness > 0:
                    fit_genomes.append(genome)
        print("Previous pool had " + str(len(fit_genomes)) + " fit genomes")
        mutated_fit = []
        for item in fit_genomes:
            mutated_fit.extend(self.mutate_genome(item, 3))
        # Select pairs to reproduce and mutate
        repr_genomes = []
        if len(fit_genomes) > 2:
            num_pairs = max(int(self.pool_size/50), int(len(fit_genomes)))
            for _ in range(num_pairs):
                g1, g2 = random.sample(fit_genomes, 2)
                offspring = (self.reproduce_genome(g1, g2, 4))
                repr_genomes.extend(offspring)
                for item in offspring:
                    repr_genomes.extend(self.mutate_genome(item, 5))
        print("New genomes from reproduction: " + str(len(repr_genomes)))
        new_genomes = []
        new_genomes.extend(fit_genomes)
        new_genomes.extend(mutated_fit)
        new_genomes.extend(repr_genomes)
        print("New genome pool size: " + str(len(new_genomes)))
        if len(new_genomes) < self.pool_size:
            pad = self.pool_size - len(new_genomes)
            print("Adding " + str(pad) + " new random genomes.")
            for _ in range(pad):
                genome = np.random.uniform(-1, 1, self.genome_size)
                new_genomes.append(genome)
        elif len(new_genomes) > self.pool_size:
            print("Taking a random sample from new genome pool.")
            new_genomes = random.sample(new_genomes, self.pool_size)
        print("Creating new genome pool")
        genome_pool = []
        for g in new_genomes:
            genome_pool.append([np.array(g), None])
        self.genome_pool = genome_pool

    def get_genome_statistics(self):
        success = 0
        fail = 0
        unused = 0
        for item in self.genome_pool:
            genome, fitness = item
            if fitness is not None:
                if fitness > 0:
                    success += 1
                else:
                    fail += 1
            else:
                unused += 1
        return success, fail, unused

    def save_genomes(self):
        with open("genome_pool.pkl", "wb") as f:
            f.write(pickle.dumps(self.genome_pool))

    def load_genomes(self):
        n = []
        with open("genome_pool.pkl", "rb") as f:
            n = pickle.load(f)
        return n


# Train the game
def msg(gs):
    msg = "Steps: " + str(steps) + "\n"
    msg += "Episode length: " + str(gs.max_episode_len) + "\n\n"
    s, f, u = gs.get_genome_statistics()
    msg += "Success: " + str(s) + " Fail: " + str(f) + " Unused: " + str(u) + "\n"
    msg += "[ "
    for s in prev_stats[-10:]:
        msg += "%.2f"%s + " "
    msg += "]\n"
    return msg

#random.seed(1)
game_space_width = 60
game_space_height = 30
num_walls = 40
num_agents = 50
num_objectives = 8
max_episode_len = 100
savedir = "save_gn"

gs = game_space(game_space_width,
                game_space_height,
                num_walls=num_walls,
                num_agents=num_agents,
                max_episode_len=max_episode_len,
                num_objectives=num_objectives,
                savedir=savedir)

print_visuals = True
steps = 0
prev_stats = []
if os.path.exists("evolution_stats.json"):
    with open("evolution_stats.json", "r") as f:
        prev_stats = json.loads(f.read())
while True:
    gs.step()
    if print_visuals == True:
        os.system('clear')
        print()
        print(gs.print_game_space())
        print()
        print(msg(gs))
        time.sleep(0.03)
    elif steps%gs.max_episode_len == 0:
        os.system('clear')
        print()
        print(msg(gs))
        print()
    if steps % gs.max_episode_len == 0:
        s, f, u = gs.get_genome_statistics()
        if u < 200:
            used = gs.pool_size - u
            sr = (s/used) * 100
            prev_stats.append(sr)
            with open("evolution_stats.json", "w") as f:
                f.write(json.dumps(prev_stats))
            gs.create_new_genome_pool()
            gs.save_genomes()
    steps += 1

