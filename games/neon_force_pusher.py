import datetime
import os
#from PIL import Image
#import gym
import numpy
from numpy.core.fromnumeric import choose
#import torch
import socket, sys
from enum import IntEnum
from .abstract_game import AbstractGame
#import random
import time
from threading import Thread
import json
from pathlib import Path
import subprocess
import threading
import random
# for storing information about the available connections

class MuZeroConfig:
    def __init__(self):
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 0  # Seed for numpy, torch and the game
        self.max_num_gpus = 0  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available

        ### Game
        self.observation_shape = (1, 40, 40)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = list(range(50))  # Fixed list of all possible actions. You should only edit the length
        self.players = list(range(1))  # List of players. You should only edit the length
        self.stacked_observations = 0  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = "self"  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class
                        # From error... "opponent" argument should be "self", "human", "expert" or "random"


        ### Self-Play
        self.num_workers = 2  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = False
        self.max_moves = 1500  # Maximum number of moves if game is not finished before
        self.num_simulations = 30  # Number of future moves self-simulated
        #self.discount = 0.997  # Chronological discount of the reward
        self.discount = 0.99  # Chronological discount of the reward
        self.temperature_threshold = 10  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time
        self.random_move_till_n_action_in_self_play = 30  # choice random moves until

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        ### Network
        self.network = "fullyconnected"  # "resnet" / "fullyconnected"
        self.support_size = 30  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))
        
        # Residual Network
        self.downsample = "CNN"  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.blocks = 16  # Number of blocks in the ResNet
        self.channels = 1  # Number of channels in the ResNet
        self.reduced_channels_reward = 6  # Number of channels in reward head
        self.reduced_channels_value = 6  # Number of channels in value head
        self.reduced_channels_policy = 6  # Number of channels in policy head
        self.resnet_fc_reward_layers = [12,24, 32, 64]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [12,24,32, 64]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [12,24,32, 64]  # Define the hidden layers in the policy head of the prediction network

        # Fully Connected Network
        self.encoding_size = 3
        self.fc_representation_layers = [64]  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [64]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [64]  # Define the hidden layers in the reward network
        self.fc_value_layers = [64]  # Define the hidden layers in the value network
        self.fc_policy_layers = [64]  # Define the hidden layers in the policy network


        ### Training
        #self.results_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "/tmp/results", os.path.basename(__file__)[:-3], datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))  # Path to store the model weights and TensorBoard logs
        self.results_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../results", os.path.basename(__file__)[:-3], datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))  # Path to store the model weights and TensorBoard logs
        self.save_model = True      # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = 100000  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 120       # Number of parts of games to train on at each training step (~ 6.1GB GPU_RAM )
        #self.batch_size = 50       # Number of parts of games to train on at each training step (~ 6.1GB GPU_RAM )
        #self.batch_size = 150       # Number of parts of games to train on at each training step  (over 9.8GB GPU_RAM needed)
        self.checkpoint_interval = 10   # Number of training steps before using the model for self-playing
        self.value_loss_weight = .25    # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.train_on_gpu = False # torch.cuda.is_available()  # Train on GPU if available

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-6  # L2 weights regularization
        self.momentum = 0.5  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.003  # Initial learning rate
        self.lr_decay_rate = .9995  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 100

        ### Replay Buffer
        self.replay_buffer_size = 35  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 55  # Number of game moves to keep for every batch element
        self.td_steps = 24  # Number of steps in the future to take into account for calculating the target value
        self.PER = False  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = .8  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.reanalyse_on_gpu = False

        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 1  # Number of seconds to wait after each played game
        self.training_delay = 1  # Number of seconds to wait after each training step
        self.ratio = .1  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it


    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        if trained_steps < 0.5 * self.training_steps:
            return 1.0
        elif trained_steps < 0.75 * self.training_steps:
            return 0.5
        else:
            return 0.25

class Commands(IntEnum):
    HELP = 0
    SHOW_ACTIONS = 1
    GET_STATE = 2
    GET_ACTIONS = 3
    GET_REWARD = 4
    _SPAM = 5 # Previously was ADVANCE_FRAME, but someone was spamming it.
    RESET = 6
    FINISH = 7
    SET_INPUT = 8
    SET_PLAYER = 9
    GET_LEVEL_SELECT_ACTIONS = 10
    ADVANCE_FRAME = 11


class NFP():
    def __init__(self, machine, port=5884):
        self.machine = machine
        self.port = port
        self._level_select_actions = None
        self._frame_time = 200
        self._actions = {}
        self.player_deaths = {0:0,1:0}
        self.player_score = {0:0,1:0}
    
    def __enter__(self):
        return self
    
    def __exit__(self):
        self.send_cmd(Commands.FINISH)

    def get_text(self, commands):
        try:
            self.connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.connection.connect((self.machine, self.port))
            self.connection.send(bytes(commands))
            data = self.connection.recv(1000)
            return data[1:].decode("ascii")
        finally:
            self.connection.close()
        

    def legal_actions(self):
        got_data = False
        while not got_data: 
            try:
                self.connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.connection.connect((self.machine, self.port))
                self.connection.send(bytes([int(Commands.GET_ACTIONS)]))
                data = self.connection.recv(100)
                data = list(data)
                if len(data) > 0:
                    got_data = True
                    if 0 in data:
                        data.remove(0)
                    return list(data)
            except:
                print("Connection issue retrying...")
            finally:
                self.connection.close()
            
    
    def get_level_select_actions(self):
        b = self.get_bytes(Commands.GET_LEVEL_SELECT_ACTIONS)
        return(list(b))

    @property
    def level_select_actions(self):
        if self._level_select_actions is None:
            self._level_select_actions = self.get_level_select_actions()
        return self._level_select_actions
    
    def advance_frame(self):
        try:

            self.connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.connection.connect((self.machine, self.port))
            self.connection.send(bytes([int(Commands.ADVANCE_FRAME)])) # move somewhere and advance frame
            
        except Exception as err:
            print(f"{err}")
            
        finally:
            self.connection.close()
        
    def set_action(self, player, action:Commands):
        self._actions[player] = action

    def move(self, moves : list):
        if isinstance(moves, int):
            moves = [moves]
        if not moves[0] == Commands.SET_INPUT:
            moves.insert(0,int(Commands.SET_INPUT))
        if len(moves) == 0:
            print("Move command missing aciton.")
        try:
            self.connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.connection.connect((self.machine, self.port))
            self.connection.send(bytes(moves)) # move somewhere and advance frame
            self._is_done = False
        finally:
            self.connection.close()

    def get_state(self):
        img_w = 40
        img_h = 40
        img_depth = 1
        got_data = False
        while not got_data:
            try:
                self.connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.connection.connect((self.machine, self.port))
                self.connection.send(bytes([int(Commands.GET_STATE)])) # get_image
                
                l_data = []
                data = self.connection.recv(img_w*img_h)
                while data:
                    l_data.extend(list(data))
                    data = self.connection.recv(img_w*img_h)
                if len(l_data) == img_w*img_h*img_depth:
                    self.data = numpy.array(l_data)
                    self.data = numpy.reshape(self.data, (img_depth, img_w,img_h))
                    if self.data.shape == (img_depth,img_w,img_h):
                        got_data = True
                        self.connection.close()
                        return self.data
                if not got_data:
                    print("stuck on get_state because no valid image.")
            except Exception as err :
                got_data= False
                print(f"Connection to {self.machine}:{self.port} failed error (get_state) retry in 10sec... {str(err)}")
                time.sleep(10.0)


    def get_bytes(self, command: Commands, expected_bytes=1000):
        got_data = False
        while not got_data:
            try:
                l_data = []
                self.connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.connection.connect((self.machine, self.port))
                self.connection.send(bytes([int(command)]))
                
                data = self.connection.recv(expected_bytes)
                if not data is None and len(data):
                    got_data = True
                    return data
            
            except :
                print("Connection error (get_bytes) retry...")
            finally:
                self.connection.close()

    def send_cmd(self, command:Commands, data=None):
        try:
            self.connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.connection.connect((self.machine, self.port))
            if data and isinstance(data, list): 
                data.insert(0, int(command))
                self.connection.send(bytes(data))
            else:
                self.connection.send(bytes([int(command)]))
        except Exception as err :
            print(f"Couldn't send command {command} {err}")
            
    @property
    def ai_reward(self):
        return int.from_bytes(self.get_bytes(Commands.GET_REWARD), byteorder="little", signed=True)

    @property
    def is_done(self):
        if self._is_done:
            print("Done is True")
        return self._is_done

    def level_run(self):
        self.is_done = False
        print("LEVEL RUNNING")

    def level_done(self):
        self._is_done=True
        for k, v in self.player_deaths.items():
            self.player_score[k] = self.player_score[k] - (100 * v)
        for k in self.player_deaths.keys():
            self.player_deaths[k] = 0
        for k in self.player_score.keys():
            self.player_score[k] = 0

        choose_level = self.get_level_select_actions()[0]
        if choose_level:
            self.send_cmd(Commands.SET_INPUT, [choose_level])

    def reset(self):
        self.connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connection.connect((self.machine, self.port))
        self.connection.send(bytes([Commands.RESET]))
        self._is_done = True
        print("RESET")
        
    def close(self):
        print("need to close")
        pass

    def set_player(self, id):
        self.send_cmd(Commands.SET_PLAYER, [id])
        if id == 0:
            self.advance_frame()

no_op_count = 0        
class FrameWorker(Thread):
    def __init__(self, machine, port, sleep_sec=.1, start_paused=False, start_delay=10):
        super().__init__()
        self.nfp = NFP(machine, port)
        self._sleep_ms = sleep_sec
        self._running = False
        self._pause = start_paused
        self.ticks = 0
        self.move_counter =0
        self.start_delay = start_delay
    @property
    def is_running(self):
        return self._running

    @property
    def is_paused(self):
        return self._pause

    def pause(self):
        self._pause = True

    def un_pause(self):
        self._pause = False

    def reset(self):
        self.move_counter = 0
        self.ticks = 0
        
    def count_move(self):
        self.move_counter +=1

    def run(self):
        self._running = True
        global no_op_count
        time.sleep(self.start_delay)
        while self._running:
            
            if self._pause:
                time.sleep(self._sleep_ms * 10)
                self._pause = False
            if not self._pause:
                self.ticks += 1
                time.sleep(self._sleep_ms)
            no_op_count +=1
        print("FrameWorker Exiting")

    def stop(self):
        self.un_pause()
        self._running = False

class GameRunner(Thread):
    def __init__(self, machine, port):
        super().__init__()
        self.machine = machine
        self.path = "./game_executables/neon_force_pushers.x86_64"
        self.port = port

    @property
    def game_port(self):
        return self.port

    def run(self):
        
        with open(f'log_{self.ident}.log', 'a')  as log:
            print(f"Running {self.path} on port {self.port}")
            self.process = subprocess.Popen([self.path, f"-m={self.port}", "-p=2"], stderr=log, stdout=log)

class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        
        self.ip = "127.0.0.1"
        first_port = 5500
        #self.ports = [5501, 5502, 5503, 5504, 5505, 5506, 5507, 5508, 5509]
        self.port =  first_port+seed
        self.game_runner = GameRunner(self.ip, self.port)
        self.game_runner.start()
        time.sleep(2)
        self.nfp_game = NFP(self.ip, self.port)
        self.frame_worker = FrameWorker(self.ip, self.port, start_paused=False)
        self.frame_worker.start()

        #self.nfp_game = NFP("192.168.1.102",5884)
        self.l_actions = []
        self.player = 0
        self.last_observation = None

    def to_play(self):
        """
        Return the current player.

        Returns:
            The current player, it should be an element of the players list in the config. 
        """
        
        return self.player

    def step(self, action):
        """
        Apply action to the game.
        
        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        observation = self.get_observation()
        no_op_count = 0
        if action in self.nfp_game.level_select_actions and not self.frame_worker.is_paused:
            self.nfp_game.level_done()
            done = True
            self.frame_worker.un_pause()
            self.player = 1 - self.player
            self.nfp_game.set_player(self.player) 
            return observation, 0, done
        else:
            try:
                self.nfp_game.move([action])
                self.frame_worker.count_move()
                done = False
                reward = float(self.nfp_game.ai_reward)

            finally:
                self.player = 1 - self.player
                self.nfp_game.set_player(self.player)

            return observation, reward, done

    def get_observation(self):
        self.last_observation = self.nfp_game.get_state()
        return self.nfp_game.get_state()

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.
        
        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.        

        Returns:
            An array of integers, subset of the action space.
        """
        return self.nfp_game.legal_actions()
        

    def reset(self):
        """
        Reset the game for a new game.
        
        Returns:
            Initial observation of the game.
        """
        if self.frame_worker.ticks > 0:
            move_ratio = float(self.frame_worker.move_counter)/float(self.frame_worker.ticks)
            print(f"Moves: {self.frame_worker.move_counter}, ticks: {self.frame_worker.ticks} ratio: {round(move_ratio, 2)}")
            self.frame_worker.reset()
        self.player = 0
        return  self.last_observation if not self.last_observation is None else self.get_observation()

    def close(self):
        """
        Properly close the game.
        """
        self.frame_worker.stop()
        self.nfp_game.__exit__()
        print('Stopping game.')
        
        if os.path.isfile("/tmp/results/game.json"):
            os.remove("/tmp/results/game.json")

    def render(self):
        """
        Display the game observation.
        """
        #self.env.render()
        #input("Press enter to take a step ")

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.

        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        actions = {
            0: "Push cart to the left",
            1: "Push cart to the right",
        }
        return f"{action_number}. does something"

    def human_to_action(self):
        """
        For multiplayer games, ask the user for a legal action
        and return the corresponding action number.

        Returns:
            An integer from the action space.
        """
        #choice = input(f"Enter the action to play for the player {self.to_play()}: ")
        #while int(choice) not in self.legal_actions():
        #    choice = input("Ilegal action. Enter another action : ")
        return 0
