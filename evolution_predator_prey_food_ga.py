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
    def __init__(self, width, height, num_walls=0, num_predators=1, num_prey=1,
                 num_food=10, max_episode_len=200, savedir="save"):
        self.savedir = savedir
        self.max_episode_len = max_episode_len
        self.width = width+2
        self.height = height+2
        self.start_walls = num_walls
        self.action_size = 5
        self.hidden_size = 32
        self.pool_size = 1000
        self.state_size = self.get_state_size()
        self.genome_size = (self.state_size*self.hidden_size) + (self.action_size*self.hidden_size)
        self.agent_types = ["predator", "prey"]
        self.starts = {"predator":2, "prey":2000}
        self.num_agents = {}
        self.num_agents["predator"] = num_predators
        self.num_agents["prey"] = num_prey
        self.num_food = num_food
        self.food = []
        self.food_id = 10000
        self.genome_pool = {}
        self.best_policies = {}
        for t in self.agent_types:
            self.best_policies[t] = []
            self.genome_pool[t] = []
            if os.path.exists(savedir + "/" + t + "_best_policies.pkl"):
                print("Loading best " + t + " policies.")
                self.best_policies[t] = self.load_best_policies(t)
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
            for index in range(self.num_agents[t]):
                self.agents[t].append(None)
        self.walls = []
        space = self.make_empty_game_space()
        self.game_space = space
        self.add_walls(self.start_walls)
        self.num_walls = len(self.walls)
        self.add_food()
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

    def add_food(self):
        locations = self.get_random_empty_space(self.num_food)
        for item in locations:
            y, x = item
            self.food.append([y, x])

    def remove_food(self, xpos, ypos):
        fi = None
        for index, item in enumerate(self.food):
            y, x = item
            if x == xpos and y == ypos:
                fi = index
        if fi is not None:
            del self.food[fi]

    def spawn_more_food(self):
        nf = len(self.food)
        if nf < self.num_food:
            mf = self.num_food - nf
            locs = self.get_random_empty_space(mf)
            for loc in locs:
                y, x = loc
                self.food.append([y, x])

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
        item = self.get_random_empty_space(1)
        ypos, xpos = item[0]
        state_size = self.get_state_size()
        action_size = self.action_size

        selectable = []
        for i, item in enumerate(self.genome_pool[atype]):
            g, f = item
            if f is None:
                selectable.append(i)
        gi = random.choice(selectable)
        item = self.genome_pool[atype][gi]
        genome, fitness = item

        self.agents[atype][index] = agent(xpos, ypos, state_size, action_size,
                                          self.hidden_size, genome, gi)

    def add_items_to_game_space(self):
        space = np.array(self.game_space)
        for t in self.agent_types:
            if len(self.agents[t]) > 0:
                for index, agent in enumerate(self.agents[t]):
                    if agent is not None:
                        space[agent.ypos][agent.xpos] = self.starts[t]+index
        for f in self.food:
            y, x = f
            space[y][x] = self.food_id
        return space

    def get_random_empty_space(self, num=1):
        space = self.add_items_to_game_space()
        empties = list(np.argwhere(space == 0))
        return random.sample(empties, num)

    def respawn_agent(self, index, atype):
        gi = self.agents[atype][index].gi
        item = self.genome_pool[atype][gi]
        genome, fitness = item
        fitness = self.agents[atype][index].fitness
        self.genome_pool[atype][gi] = [genome, fitness]
        self.create_new_agent(index, atype)
        state = self.get_agent_state(index, atype)
        self.agents[atype][index].state = state

    def move_forward(self, index, action, atype):
        got_food = False
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
        if atype == "prey":
            if space_val == self.food_id:
                got_food = True
                self.remove_food(newx, newy)
                self.spawn_more_food()
        return newx, newy, space_val, got_food

    def update_agent_success(self, atype, index):
        s = self.agents[atype][index].episode_steps
        self.agents[atype][index].last_success = s

    def move_agent(self, index, action, atype):
        done = False
        died = False
        if action == 0: # do nothing
            pass
        else:
            newx, newy, space_val, got_food = self.move_forward(index, action, atype)
            if got_food == True:
                self.agents[atype][index].fitness += 10
                self.update_agent_success(atype, index)
            if atype == "predator":
                if space_val >= self.starts["prey"] and space_val < self.food_id:
                    prey_index = self.get_prey_at_position(newx, newy)
                    adjacent_predator = self.get_adjacent_friend_count(index, atype)
                    adjacent_prey = self.get_adjacent_friend_count(prey_index, "prey")
                    # Adjacent prey lower the chance of capture
                    # and increase the chance the predator will die instead
                    # Adjacent predators negate prey deflect chance
                    deflect_chance = (0.3*adjacent_prey) - (0.3*adjacent_predator)
                    if random.random() > deflect_chance:
                        self.respawn_agent(prey_index, "prey")
                        self.agents[atype][index].fitness += 10
                        self.update_agent_success(atype, index)
                    else:
                        died = True
                        self.agents["prey"][prey_index].fitness += 10
                        self.update_agent_success("prey", prey_index)
                        self.respawn_agent(index, atype)
        state = self.get_agent_state(index, atype)
        self.agents[atype][index].state = state
        if died == False:
            self.agents[atype][index].fitness += 1
            self.agents[atype][index].episode_steps += 1
            ls = self.agents[atype][index].last_success
            if ls is not None:
                s = self.agents[atype][index].episode_steps
                if (s - ls) > self.max_episode_len:
                    self.respawn_agent(index, atype)
            elif self.agents[atype][index].episode_steps >= self.max_episode_len:
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
        if tile == 1:
            return 1
        elif tile >= self.starts["predator"] and tile < self.starts["prey"]:
            return 2
        elif tile >= self.starts["prey"] and tile < self.food_id:
            return 3
        elif tile == self.food_id:
            return 4

    def distance(self, xa, ya, xb, yb):
        dst = distance.euclidean([xa, ya], [xb, yb])
        return dst

    def get_directions(self, xpos, ypos, positions, num):
        ind = {}
        for index, item in enumerate(positions):
            ey, ex = item
            dist = self.distance(ex, ey, xpos, ypos)
            ind[index] = dist
        nearest_indices = []
        for index, dist in sorted(ind.items(), key=operator.itemgetter(1),reverse=False):
            nearest_indices.append(index)
            if len(nearest_indices) >= num:
                break
        directions = []
        for index in nearest_indices:
            item = positions[index]
            ey, ex = item
            xoff = ex-xpos
            yoff = ey-ypos
            directions.append([xoff,yoff])
        return directions

    def get_nearest_enemy_directions(self, xpos, ypos, atype, num):
        sp = self.add_items_to_game_space()
        positions = []
        if atype == "predator":
            positions = np.argwhere((sp>=self.starts["prey"]) & (sp<self.food_id))
        else:
            positions = np.argwhere((sp>=self.starts["predator"]) & (sp<self.starts["prey"]))
        offsets = self.get_directions(xpos, ypos, positions, num)
        return offsets

    def get_nearest_friend_directions(self, xpos, ypos, atype, num):
        sp = self.add_items_to_game_space()
        positions = []
        if atype == "prey":
            positions = np.argwhere((sp>=self.starts["prey"]) & (sp<self.food_id))
        else:
            positions = np.argwhere((sp>=self.starts["predator"]) & (sp<self.starts["prey"]))
        offsets = self.get_directions(xpos, ypos, positions, num)
        return offsets

    def get_nearest_food_directions(self, xpos, ypos, num):
        sp = self.add_items_to_game_space()
        positions = np.argwhere(sp==self.food_id)
        offsets = self.get_directions(xpos, ypos, positions, num)
        return offsets

    def get_state_size(self):
        state_size = 32
        return state_size

    def add_directions_to_state(self, state, directions):
        if directions is not None:
            for item in directions:
                yp, xp = item
                state.append(xp)
                state.append(yp)
        return state

    def get_adjacent_friend_count(self, index, atype):
        space = self.add_items_to_game_space()
        xpos = self.agents[atype][index].xpos
        ypos = self.agents[atype][index].ypos
        os = [-1, 0, 1]
        offsets = []
        for x in os:
            for y in os:
                if x == 0 and y == 0:
                    continue
                offsets.append([x, y])
        count = 0
        for i in offsets:
            oy, ox = i
            tile = self.get_tile_val(space[ypos+oy][xpos+ox], atype)
            if atype == "predator" and tile == 2:
                count += 1
            if atype == "prey" and tile == 3:
                count += 1
        return count

    def make_small_state(self, index, atype):
        xpos = self.agents[atype][index].xpos
        ypos = self.agents[atype][index].ypos
        tiles = []
        if atype == "predator":
            directions = self.get_nearest_enemy_directions(xpos, ypos, atype, 2)
            tiles = self.add_directions_to_state(tiles, directions)
            directions = self.get_nearest_friend_directions(xpos, ypos, atype, 2)
            tiles = self.add_directions_to_state(tiles, directions)
        else:
            directions = self.get_nearest_enemy_directions(xpos, ypos, atype, 1)
            tiles = self.add_directions_to_state(tiles, directions)
            directions = self.get_nearest_food_directions(xpos, ypos, 1)
            tiles = self.add_directions_to_state(tiles, directions)
            directions = self.get_nearest_friend_directions(xpos, ypos, atype, 2)
            tiles = self.add_directions_to_state(tiles, directions)

        space = self.add_items_to_game_space()
        os = [-2, -1, 0, 1, 2]
        offsets = []
        for x in os:
            for y in os:
                if x == 0 and y == 0:
                    continue
                offsets.append([x, y])

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

    def get_min_best_fitness(self, atype):
        best_len = len(self.best_policies[atype])
        if best_len < self.pool_size:
            return self.max_episode_len
        min_val = None
        min_index = None
        for index, item in enumerate(self.best_policies[atype]):
            genome, fitness = item
            if min_val is None:
                min_val = fitness
                min_index = index
            if fitness < min_val:
                min_val = fitness
                min_index = index
        return min_val, min_index

    def replace_best_fitness_entry(self, atype, index, genome, fitness):
        self.best_policies[atype][index] = ([genome, fitness])

    def trim_best_policies(self, atype):
        bpl = len(self.best_policies[atype])
        if bpl > self.pool_size:
            while len(self.best_policies[atype]) > self.pool_size:
                val, index = self.get_min_best_fitness(atype)
                del self.best_policies[atype][index]

    def add_best_fitness_entry(self, atype, genome, fitness):
        self.trim_best_policies(atype)
        if fitness < self.max_episode_len:
            return
        best_len = len(self.best_policies[atype])
        if best_len >= self.pool_size:
            mv, mi = self.get_min_best_fitness(atype)
            if fitness > mv:
                self.replace_best_fitness_entry(atype, mi, genome, fitness)
        else:
            self.best_policies[atype].append([genome, fitness])

    def get_best_policy_stats(self, atype):
        f = []
        l = len(self.best_policies[atype])
        for item in self.best_policies[atype]:
            genome, fitness = item
            f.append(fitness)
        if len(f) > 0:
            return np.mean(f), l
        else:
            return 0, 0

    def get_best_policies(self, atype, num):
        vi = {}
        for index, item in enumerate(self.best_policies[atype]):
            genome, fitness = item
            vi[index] = fitness
        ret = []
        for index, fitness in sorted(vi.items(), key=operator.itemgetter(1),reverse=True):
            genome = self.best_policies[atype][index][0]
            ret.append(genome)
            if len(ret) >= num:
                break
        return ret

    def get_best_genomes(self, atype, threshold):
        pool = self.genome_pool[atype]
        fit_genomes = []
        vi = {}
        for index, item in enumerate(pool):
            genome, fitness = item
            if fitness is not None:
                if fitness > self.max_episode_len:
                    vi[index] = fitness
        count = 0
        for index, fitness in sorted(vi.items(), key=operator.itemgetter(1),reverse=True):
            genome = pool[index][0]
            fit_genomes.append(genome)
            self.add_best_fitness_entry(atype, genome, fitness)
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
        threshold = 0.20
        fit_genomes = self.get_best_genomes(atype, threshold)
        best_policies = []
        if len(self.best_policies[atype]) > 50:
            best_policies = self.get_best_policies(atype, 50)
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
                offspring = (self.reproduce_genome(g1, g2, 3))
                repr_genomes.extend(offspring)
                for item in offspring:
                    repr_genomes.extend(self.mutate_genome(item, 3))
        msg += "New genomes from reproduction: " + str(len(repr_genomes)) + "\n"
        new_genomes.extend(fit_genomes)
        new_genomes.extend(best_policies)
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
                if fitness > self.max_episode_len:
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
        if len(f) > 0:
            return np.mean(f), max(f)
        else:
            return 0,0

    def save_best_policies(self, atype):
        with open(self.savedir + "/" + atype + "_best_policies.pkl", "wb") as f:
            f.write(pickle.dumps(self.best_policies[atype]))

    def load_best_policies(self, atype):
        n = []
        with open(self.savedir + "/" + atype + "_best_policies.pkl", "rb") as f:
            n = pickle.load(f)
        return n

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
        elif item >= self.starts["prey"] and item < self.food_id:
            return "\x1b[1;35;40m" + "x" + "\x1b[0m"
        elif item == self.food_id:
            return "\x1b[1;31;40m" + "o" + "\x1b[0m"
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
    colors = {"predator":"32;40", "prey":"35;40"}
    msg = "Steps: " + str(steps) + " Episode length: " + str(gs.max_episode_len) 
    msg += " Genome size: " + str(gs.genome_size) 
    msg += " Pool size: " + str(gs.pool_size) + "\n\n"
    for t in gs.agent_types:
        s, u = gs.get_genome_statistics(t)
        f, m = gs.get_genome_fitness(t)
        pf, pn = gs.get_best_policy_stats(t)
        msg += "\x1b[1;"+colors[t]+"m"+t+"\x1b[0m"
        msg += ": Success: " + str(s) + " Unused: " + str(u) 
        msg += " Fitness: " + "%.2f"%f + " Max: " + str(m) + "\n"
        msg += "Best policy fitness: " + "%.2f"%pf + " Num: " + str(pn) + "\n"
        msg += "[ "
        p = prev_stats[t]
        for s in p[-10:]:
            msg += "%.2f"%s + " "
        msg += "]\n\n"
    return msg

random.seed(1337)

game_space_width = 60
game_space_height = 30
num_walls = 40
num_predators = 4
num_prey = 40
num_food = 50
max_episode_len = 50
savedir = "predator_prey_food_save"
if not os.path.exists(savedir):
    os.makedirs(savedir)

gs = game_space(game_space_width,
                game_space_height,
                num_walls=num_walls,
                num_predators=num_predators,
                num_prey=num_prey,
                num_food=num_food,
                max_episode_len=max_episode_len,
                savedir=savedir)

print_visuals = False
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
        time.sleep(0.03)
    elif steps%gs.max_episode_len == 0:
        os.system('clear')
        print()
        print(msg(gs))
        print(prev_train_msg)
        print()
    if steps % int(gs.max_episode_len/2) == 0:
        for t in gs.agent_types:
            s0, u0 = gs.get_genome_statistics(t)
            if u0 < 70:
                prev_train_msg = ""
                for tt in gs.agent_types:
                    f, m = gs.get_genome_fitness(tt)
                    prev_stats[tt].append(f)
                    with open(savedir + "/evolution_stats_"+tt+".json", "w") as f:
                        f.write(json.dumps(prev_stats[tt]))
                    prev_train_msg += gs.create_new_genome_pool(tt)
                    gs.save_genomes(tt)
                    gs.save_best_policies(tt)
    steps += 1
