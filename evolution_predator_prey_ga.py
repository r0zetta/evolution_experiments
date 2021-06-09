import random, sys, time, os, json, re, pickle, operator, math
import numpy as np
from collections import Counter, deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
from scipy.spatial import distance

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
    def __init__(self, x, y, state_size, action_size, hidden_size, genome, gi):
        self.xpos = x
        self.ypos = y
        self.fitness = 0
        self.gi = gi
        fc1_weights = torch.Tensor(np.reshape(genome[:state_size*hidden_size],
                                              (hidden_size, state_size)))
        fc2_weights = torch.Tensor(np.reshape(genome[state_size*hidden_size:],
                                              (action_size, hidden_size)))
        self.model = GN_model(state_size, action_size, hidden_size, fc1_weights, fc2_weights)
        self.state = None
        self.episode_steps = 0
        self.last_success = None

    def get_action(self):
        return self.model.get_action(self.state)

class game_space:
    def __init__(self, width, height, num_walls=0, num_agents=1, max_episode_len=200, 
                 savedir="save"):
        self.savedir = savedir
        self.max_episode_len = max_episode_len
        self.num_agents = num_agents
        self.width = width+2
        self.height = height+2
        self.start_walls = num_walls
        self.action_size = 5
        self.hidden_size = 32
        self.pool_size = 500
        self.state_size = self.get_state_size()
        self.genome_size = (self.state_size*self.hidden_size) + (self.action_size*self.hidden_size)
        self.agent_types = ["predator", "prey"]
        self.starts = {"predator":2, "prey":2000}
        self.genome_pool = {}
        for t in self.agent_types:
            self.genome_pool[t] = []
            if os.path.exists(savedir + "/" + t + "_genome_pool.pkl"):
                print("Loading " + t + " genomes.")
                self.genome_pool[t] = self.load_genomes(t)
            else:
                print("Creating " + t + " genomes.")
                self.genome_pool[t] = self.make_genomes()
        self.reset()

    def reset(self):
        self.agents = {}
        for t in self.agent_types:
            self.agents[t] = []
            for index in range(self.num_agents):
                self.agents[t].append(None)
        self.walls = []
        space = self.make_empty_game_space()
        self.game_space = space
        self.add_walls(self.start_walls)
        self.num_walls = len(self.walls)
        self.initial_game_space = np.array(self.game_space)
        for t in self.agent_types:
            for index, agent in enumerate(self.agents[t]):
                self.create_new_agent(index, t)
        for t in self.agent_types:
            for index, agent in enumerate(self.agents[t]):
                state = self.get_agent_state(index, t)
                self.agents[t][index].state = state

    def step(self):
        for t in self.agent_types:
            for index in range(len(self.agents[t])):
                action = gs.agents[t][index].get_action()
                self.move_agent(index, action, t)

    def make_empty_game_space(self):
        space = np.zeros((self.height, self.width), dtype=int)
        for n in range(self.width):
            space[0][n] = 1
            space[1][n] = 1
            space[self.height-1][n] = 1
            space[self.height-2][n] = 1
        for n in range(self.height):
            space[n][0] = 1
            space[n][1] = 1
            space[n][self.width-1] = 1
            space[n][self.width-2] = 1
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
        while added < num:
            item = self.get_random_empty_space(1)
            ypos, xpos = item[0]
            self.game_space[ypos][xpos] = 1
            self.walls.append([ypos, xpos])
            for n in range(50):
                move = random.randint(0,3)
                if move == 0:
                    xpos = max(0, xpos-1)
                elif move == 1:
                    xpos = min(self.width-1, xpos+1)
                elif move == 2:
                    ypos = max(0, ypos-1)
                elif move == 3:
                    ypos = min(self.height-1, ypos+1)
                if self.game_space[ypos][xpos] == 0:
                    added += 1
                self.game_space[ypos][xpos] = 1
                self.walls.append([ypos, xpos])
                if added >= num:
                    break

    def create_new_agent(self, index, atype):
        #print("Create new agent " + str(atype) + " " + str(index))
        item = self.get_random_empty_space(1)
        ypos, xpos = item[0]
        state_size = self.get_state_size()
        action_size = self.action_size
        genome = None
        gi = 0
        for i, item in enumerate(self.genome_pool[atype]):
            genome, fitness = item
            if fitness is not None:
                gi = i
                break

        self.agents[atype][index] = agent(xpos, ypos, state_size, action_size,
                                          self.hidden_size, genome, gi)

    def add_items_to_game_space(self):
        space = np.array(self.game_space)
        for t in self.agent_types:
            if len(self.agents[t]) > 0:
                for index, agent in enumerate(self.agents[t]):
                    if agent is not None:
                        space[agent.ypos][agent.xpos] = self.starts[t]+index
        return space

    def get_random_empty_space(self, num=1):
        space = self.add_items_to_game_space()
        empties = list(np.argwhere(space == 0))
        return random.sample(empties, num)

    def respawn_agent(self, index, atype):
        #print("Respawn " + str(atype) + " " + str(index))
        gi = self.agents[atype][index].gi
        item = self.genome_pool[atype][gi]
        genome, fitness = item
        fitness = self.agents[atype][index].fitness
        self.genome_pool[atype][gi] = [genome, fitness]
        self.create_new_agent(index, atype)
        state = self.get_agent_state(index, atype)
        self.agents[atype][index].state = state

    def move_forward(self, index, action, atype):
        xpos = None
        ypos = None
        xpos = self.agents[atype][index].xpos
        ypos = self.agents[atype][index].ypos
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
            self.agents[atype][index].xpos = newx
            self.agents[atype][index].ypos = newy
        return newx, newy, space_val

    def move_agent(self, index, action, atype):
        done = False
        if action == 0: # do nothing
            pass
        else:
            newx, newy, space_val = self.move_forward(index, action, atype)
            if atype == "predator":
                if space_val >= self.starts["prey"]:
                    # Predator killed a prey, increase fitness
                    self.agents[atype][index].fitness += 1
                    s = self.agents[atype][index].episode_steps
                    self.agents[atype][index].last_success = s
                    # Respawn the prey agent
                    prey_index = self.get_prey_at_position(newx, newy)
                    #print("Predator " + str(index) + " killed prey " + str(prey_index))
                    prey_fitness = self.agents["prey"][prey_index].fitness
                    #print("Prey fitness: " + str(prey_fitness))
                    self.respawn_agent(prey_index, "prey")
        state = self.get_agent_state(index, atype)
        if atype == "prey":
            self.agents[atype][index].fitness += 1
            self.agents[atype][index].state = state
            self.agents[atype][index].episode_steps += 1
            if self.agents[atype][index].episode_steps >= self.max_episode_len:
                #print("Prey " + str(index) + " hit max episode len")
                self.respawn_agent(index, atype)
        else:
            self.agents[atype][index].state = state
            self.agents[atype][index].episode_steps += 1
            ls = self.agents[atype][index].last_success
            # Predator continues to live if it feeds often enough
            if ls is not None:
                s = self.agents[atype][index].episode_steps
                if (s - ls) > self.max_episode_len/2:
                    self.respawn_agent(index, atype)
            elif self.agents[atype][index].episode_steps >= self.max_episode_len/2:
                #print("Predator " + str(index) + " hit max episode len")
                self.respawn_agent(index, atype)


    def get_prey_at_position(self, xpos, ypos):
        for index, agent in enumerate(self.agents["prey"]):
            if agent.xpos == xpos and agent.ypos == ypos:
                return index
        return None

    def get_agent_state(self, index, atype):
        return self.make_small_state(index, atype)

    def get_tile_val(self, tile, atype):
        if tile == 0:
            return 0
        if atype == "predator":
            if tile >= self.starts["prey"]:
                return 1
            else:
                return -1
        else:
            if tile >= self.starts["predator"] and tile < self.starts["prey"]:
                return -1
            else:
                return 1

    def distance(self, xa, ya, xb, yb):
        dst = distance.euclidean([xa, ya], [xb, yb])
        return dst

    def get_nearest_enemy_offset(self, xpos, ypos, atype):
        sp = self.add_items_to_game_space()
        enemy_positions = []
        if atype == "predator":
            enemy_positions = np.argwhere(sp>=self.starts["prey"])
        else:
            enemy_positions = np.argwhere((sp>=self.starts["predator"]) & (sp<self.starts["prey"]))
        ind = {}
        for index, item in enumerate(enemy_positions):
            ey, ex = item
            dist = self.distance(ex, ey, xpos, ypos)
            ind[index] = dist
        nearest_index = None
        for index, dist in sorted(ind.items(), key=operator.itemgetter(1),reverse=False):
            nearest_index = index
            break
        item = enemy_positions[nearest_index]
        ey, ex = item
        xoff = 0
        yoff = 0
        if ex < xpos:
            xoff = -1
        elif ex > xpos:
            xoff = 1
        if ey < ypos:
            yoff = -1
        elif ey > ypos:
            yoff = 1
        return xoff, yoff

    def get_state_size(self):
        state_size = 10
        return state_size

    def make_small_state(self, index, atype):
        xpos = self.agents[atype][index].xpos
        ypos = self.agents[atype][index].ypos
        xoff, yoff = self.get_nearest_enemy_offset(xpos, ypos, atype)

        space = self.add_items_to_game_space()
        os = [-1, 0, 1]
        offsets = []
        for x in os:
            for y in os:
                if x == 0 and y == 0:
                    continue
                offsets.append([x, y])

        tiles = [xoff, yoff]
        for i in offsets:
            oy, ox = i
            tile = self.get_tile_val(space[ypos+oy][xpos+ox], atype)
            tiles.append(tile)
        state = tiles
        state = np.array(state)
        state = torch.FloatTensor(state)
        state = state.unsqueeze(0)
        return state

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

    def get_best_genomes(self, pool, threshold):
        fit_genomes = []
        vi = {}
        for index, item in enumerate(pool):
            genome, fitness = item
            if fitness is not None:
                if fitness > 0:
                    vi[index] = fitness
        count = 0
        for index, fitness in sorted(vi.items(), key=operator.itemgetter(1),reverse=True):
            genome = pool[index][0]
            fit_genomes.append(genome)
            count += 1
            if count > int(len(vi) * threshold):
                break
        return fit_genomes

    def create_new_genome_pool(self, atype):
        # Get genomes from previous pool that had positive fitness
        new_genomes = []
        msg = ""
        pool = self.genome_pool[atype]
        # Grab unused genomes
        for item in pool:
            genome, fitness = item
            if fitness is None:
                new_genomes.append(genome)
        threshold = 0.25
        fit_genomes = self.get_best_genomes(pool, threshold)
        msg += atype + ": Previous pool had " + str(len(fit_genomes)) + " fit genomes.\n"
        mutated_fit = []
        for item in fit_genomes:
            mutated_fit.extend(self.mutate_genome(item, 3))
        msg += "New genomes from mutations: " + str(len(mutated_fit)) + "\n"
        # Select pairs to reproduce and mutate
        repr_genomes = []
        if len(fit_genomes) > 2:
            num_pairs = min(int(self.pool_size/50), int(len(fit_genomes)))
            for _ in range(num_pairs):
                g1, g2 = random.sample(fit_genomes, 2)
                offspring = (self.reproduce_genome(g1, g2, 4))
                repr_genomes.extend(offspring)
                for item in offspring:
                    repr_genomes.extend(self.mutate_genome(item, 5))
        msg += "New genomes from reproduction: " + str(len(repr_genomes)) + "\n"
        new_genomes.extend(fit_genomes)
        new_genomes.extend(mutated_fit)
        new_genomes.extend(repr_genomes)
        msg += "New genome pool size: " + str(len(new_genomes)) + "\n"
        if len(new_genomes) < self.pool_size:
            pad = self.pool_size - len(new_genomes)
            msg += "Adding " + str(pad) + " new random genomes." + "\n"
            for _ in range(pad):
                genome = np.random.uniform(-1, 1, self.genome_size)
                new_genomes.append(genome)
        elif len(new_genomes) > self.pool_size:
            msg += "Taking a random sample from new genome pool." + "\n"
            new_genomes = random.sample(new_genomes, self.pool_size)
        genome_pool = []
        for g in new_genomes:
            genome_pool.append([np.array(g), None])
        self.genome_pool[atype] = genome_pool
        msg += "\n"
        return msg

    def get_genome_statistics(self, atype):
        success = 0
        unused = 0
        pool = self.genome_pool[atype]
        for item in pool:
            genome, fitness = item
            if fitness is not None:
                if atype == "predator":
                    if fitness > 0:
                        success += 1
                else:
                    if fitness > self.max_episode_len*0.95:
                        success += 1
            else:
                unused += 1
        return success, unused

    def get_genome_fitness(self, atype):
        f = []
        pool = self.genome_pool[atype]
        for item in pool:
            genome, fitness = item
            if fitness is not None:
                f.append(fitness)
        return np.mean(f)

    def save_genomes(self, atype):
        with open(self.savedir + "/" + atype + "_genome_pool.pkl", "wb") as f:
            f.write(pickle.dumps(self.genome_pool[atype]))

    def load_genomes(self, atype):
        n = []
        with open(self.savedir + "/" + atype + "_genome_pool.pkl", "rb") as f:
            n = pickle.load(f)
        return n



    def get_printable(self, item):
        if item == 1:
            return "\x1b[1;37;40m" + "░" + "\x1b[0m"
        elif item >= self.starts["predator"] and item < self.starts["prey"]:
            return "\x1b[1;32;40m" + "x" + "\x1b[0m"
        elif item >= self.starts["prey"]:
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


# Train the game
def msg(gs):
    msg = "Steps: " + str(steps) + " Episode length: " + str(gs.max_episode_len) + "\n\n"
    for t in gs.agent_types:
        s, u = gs.get_genome_statistics(t)
        f = gs.get_genome_fitness(t)
        msg += t + ": Success: " + str(s) + " Unused: " + str(u) 
        msg += " Fitness: " + "%.2f"%f + "\n"
        msg += "[ "
        p = prev_stats[t]
        for s in p[-10:]:
            msg += "%.2f"%s + " "
        msg += "]\n\n"
    return msg

#random.seed(1)
game_space_width = 60
game_space_height = 30
num_walls = 100
num_agents = 50
max_episode_len = 100
savedir = "predator_prey_save"
if not os.path.exists(savedir):
    os.makedirs(savedir)

gs = game_space(game_space_width,
                game_space_height,
                num_walls=num_walls,
                num_agents=num_agents,
                max_episode_len=max_episode_len,
                savedir=savedir)

print_visuals = True
steps = 0
prev_stats = {}
for t in gs.agent_types:
    prev_stats[t] = []
    if os.path.exists(savedir + "/evolution_stats_"+t+".json"):
        with open(savedir + "/evolution_stats_"+t+".json", "r") as f:
            prev_stats[t] = json.loads(f.read())
prev_train_msg = ""
while True:
    gs.step()
    if print_visuals == True:
        os.system('clear')
        print()
        print(gs.print_game_space())
        print()
        print(msg(gs))
        #time.sleep(0.03)
    elif steps%gs.max_episode_len == 0:
        os.system('clear')
        print()
        print(msg(gs))
        print(prev_train_msg)
        print()
    if steps % int(gs.max_episode_len/2) == 0:
        for t in gs.agent_types:
            s0, u0 = gs.get_genome_statistics(t)
            if u0 < 30:
                prev_train_msg = ""
                for tt in gs.agent_types:
                    s, u = gs.get_genome_statistics(tt)
                    used = gs.pool_size - u
                    if used > 0:
                        sr = (s/used) * 100
                    else:
                        sr = 100
                    prev_stats[tt].append(sr)
                    with open(savedir + "/evolution_stats_"+tt+".json", "w") as f:
                        f.write(json.dumps(prev_stats[tt]))
                    prev_train_msg += gs.create_new_genome_pool(tt)
                    gs.save_genomes(tt)
    steps += 1

