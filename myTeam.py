# baselineTeam.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import util

from captureAgents import CaptureAgent
from game import Directions
from util import nearestPoint


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########


class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        # features = util.Counter()
        # successor = self.get_successor(game_state, action)
        # features['successor_score'] = self.get_score(successor)
        # return features

        

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        food_to_defend = self.get_food_you_are_defending(successor).as_list()   # food to defend
        food_to_eat = self.get_food(successor).as_list()                        # food to eat

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # self.getScore(successor)
        features['successor_score'] = -len(food_to_eat)

        # Distance to the nearest food to eat
        distances_to_food = []

        for f in food_to_eat:
            dist = self.get_maze_distance(successor.get_agent_state(self.index).get_position(), f)
            distances_to_food.append(dist)
        
        features['distance_to_food'] = min(distances_to_food)

        # Distance to the nearest power capsule to eat
        
        if self.get_capsules(game_state):
            distances_to_capsule = []
            capsules_to_eat = self.get_capsules(game_state)

            for c in capsules_to_eat:
                dist = self.get_maze_distance(successor.get_agent_state(self.index).get_position(), c)
                distances_to_capsule.append(dist)
            
            features['distance_to_capsule'] = min(distances_to_capsule)
        
        # Check if the agent is a Pacman (in enemy territory) or a ghost (in safe territory)
        # features['is_pacman'] = int(game_state.get_agent_state(self.index).is_pacman)

        # Distances to the nearest ghost and pacman opponents
        distances_to_ghosts = []
        distances_to_pacmans = []

        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        defenders = [a for a in enemies if not(a.is_pacman) and a.get_position() is not None]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        # features['num_invaders'] = len(invaders)
        features['num_defenders'] = len(defenders)

        if len(defenders) > 0:
            distances_to_ghosts = [self.get_maze_distance(my_pos, a.get_position()) for a in defenders]
            features['defender_distance'] = min(distances_to_ghosts)
            
        # if len(invaders) > 0:
        #     distances_to_pacmans = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
        #     features['invader_distance'] = min(distances_to_pacmans)
        
        features['num_carrying'] = successor.get_agent_state(self.index).num_carrying
        features['scared_timer'] = successor.get_agent_state(self.index).scared_timer
        features['num_returned'] = successor.get_agent_state(self.index).num_returned
        
        """# Check the distance to get to safety in our side of the grid
        distance_to_safety = []

        for x in 
            distances_to_safety.append(self.get_maze_distance(successor.get_agent_state(self.index).get_position(), x)
        features['distance_to_safety'] = min(distance_to_safety)"""

        return features
        
    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'distance_to_food': -1, 'distance_to_capsule': -1, 'num_defenders': -1, 'defender_distance': -1, 'num_carrying': 1, 'scared_timer': 1, 'num_returned': 3}


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}
