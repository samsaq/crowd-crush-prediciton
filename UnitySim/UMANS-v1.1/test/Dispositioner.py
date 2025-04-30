# UMANS: Unified Microscopic Agent Navigation Simulator
# MIT License
# Copyright (C) 2018-2020  Inria Rennes Bretagne Atlantique - Rainbow - Julien Pettré
# 
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject
# to the following conditions:
# 
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# 
# Contact: crowd_group@inria.fr
# Website: https://project.inria.fr/crowdscience/
# See the file AUTHORS.md for a list of all contributors.

#UTILS FUNCTIONS: generating parametrized random dispositions for the Generator
import math
import random

#default datas
number_of_agents = 50
min_dist = 1
#default start area
# X
start_x_range = [-6,-4]
# Y
start_y_range = [-10,10]
#default goal area
#X
goal_x_range = [4,6]
#Y
goal_y_range = [-10, 10]


def generate_random_position2D(x_range, y_range):
    new_x = random.uniform(x_range[0], x_range[1])
    new_y = random.uniform(y_range[0], y_range[1])
    return [new_x, new_y]

def distance2D(pointA, pointB):
    distance = math.sqrt( ((pointA[0]-pointB[0])**2)+((pointA[1]-pointB[1])**2) )
    return distance

def check_position2D(new_pos,list_of_positions,min_dist):
    for pos_i in list_of_positions:
        if(distance2D(pos_i,new_pos) <= min_dist):
            return False
    return True

def generate_random_value(value_range): #1 to 1.6
    return random.uniform(value_range[0], value_range[1])




# main functions

def generate_n_random_velocities_in_range(n_vel, val_range):
    generated_velocities = [generate_random_value(val_range)]
    for x in range(1,n_vel):
        generated_velocities.append(generate_random_value(val_range))
    return generated_velocities

def generate_n_positions_in_range(n_pos,x_range, y_range, min_dist):
    generated_positions = [generate_random_position2D(x_range,y_range)]
    for x in range(1,n_pos):
        new_pos = generate_random_position2D(x_range,y_range)
        while_iterations = 0
        while(not check_position2D(new_pos,generated_positions,min_dist)):
            new_pos = generate_random_position2D(x_range,y_range)
            if(while_iterations > 500):
                break
        generated_positions.append(new_pos)
    return generated_positions


def translate_positions(positions, vector):
    translated_positions = list()
    for pos in positions:
        new_pos = [pos[0]+vector[0], pos[1]+vector[1]]
        translated_positions.append(new_pos)
    return translated_positions
    

def join_positions(positions_A, positions_B):
    return positions_A + positions_B
