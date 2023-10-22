# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
This file includes the implementation of the Racetrack environment (as described
in Sutton & Barto's "Reinforcement Learning - An introduction" book ), and the
implementation of three different Temporal Different algorithms (i.e., SARSA,
Q-Learning and Expected SARSA) for addressing the Control problem.

Author(s):  Enrico Stefanel (enrico.stefanel@studenti.units.it)
Date:       2023-10-03
"""

import pickle # To save Racetrack objects on filesystem (e.g., to avoid having
              # to run the Control loop every time the notebook is opened)
import logging
from tqdm import tqdm

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap

from utils.WooAmanatides import WooLineAlgorithm # Algorithm for line intersection check

logger = logging.getLogger('racetrack')
logging.basicConfig(
    level=logging.WARNING, # Set the loggin level to WARNING
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)


class Track():
    """
    This class implements a racetrack environment. The track is represented
    as a matrix of integers, where 0 represents a non-track cell, 1 represents
    a starting cell, 2 represents a road cell, and 3 represents an ending cell.
    The track is loaded from a text file, and the class provides methods to
    plot the track, and to check if a line segment intersects the track.
    """

    def __init__(self, track_type='Track1'):
        """
        Create a Track object, given the track type.
        Input(s):   track_type:     the type of track to be loaded. Default is
                                    'Track1'.
        Output(s):  None
        Example:    track = Track(track_type='Track1')
        """

        # Check if track_type is one of the available files presents
        # in the utils folder
        assert track_type in ['Track1', 'Track2'], "The track type must be one of Track1 or Track2."

        # Save the track name
        self.name = track_type

        # Load the track shape from filesystem
        self.track = np.loadtxt(f'./data/{track_type}.txt')

        # Vetically flip the track matrix so that it is possible
        # to move from the bottom to the top by positively incrmenting
        # the vertical component of the velocity
        self.track = np.flipud(self.track)

        # Save the track dimensions
        self.height, self.width = self.track.shape
        self.shape = (self.width, self.height)

        # Mark the start and end line, so that start tiles
        # have value 1, road tiles have value 1 and end tiles
        # have value 3. Non-track tiles have value 0.
        self.track[: ,  :] = np.where(self.track[:, :] == 1, 2, 0)
        self.track[0,  :] = np.where(self.track[0, :] == 2, 1, 0)
        self.track[: , -1] = np.where(self.track[:, -1] == 2, 3, 0)

        # Save starting and ending positions
        self.starting_positions = np.column_stack(np.where(self.track == 1)[::-1])
        self.ending_positions = np.column_stack(np.where(self.track == 3)[::-1])

        # Construct the rewards matrix
        self.rewards = np.where(self.track == 0, 0, -1)

    def render(self):
        """
        Render the track.
        Input(s):   None
        Output(s):  None
        Example:    track.render()
        """

        fig, (ax1) = plt.subplots(1, 1, figsize=(7, 5))

        self._plot_track_boundary(ax1)
        self._plot_track(ax1)

        plt.show()

    def _plot_track_boundary(self, ax):
        """
        Add the track boundary to the specified Pyplot axis.
        Input(s):   ax:     the axis to which the track boundary is added
        Output(s):  None
        Example:    track._plot_track_boundary(ax)
        """

        ax.set_xticks(np.arange(-1, self.width, 1)+.5)
        ax.set_xticklabels([])
        ax.set_yticks(np.arange(-1, self.height, 1)+.5)
        ax.set_yticklabels([])
        ax.tick_params(left = False, bottom = False)
        ax.grid(True, axis='both', lw=.5, c='black')
        ax.tick_params(axis='x', length=0)
        ax.set_aspect('equal')

    def _plot_track(self, ax):
        """
        Add the track to the specified Pyplot axis.
        Input(s):   ax:     the axis to which the track is added
        Output(s):  None
        Example:    track._plot_track(ax)
        """

        # Define the colormap for the track
        cmapmine = ListedColormap(['black', 'green', 'white', 'red'], N=4)

        ax.imshow(
            self.track, cmap=cmapmine, vmin=0, vmax=3,
            origin='lower'
            )


class TrackEnv():
    """
    This class implements the environment of the Track model, where the matrix
    (state,action) is known, but the model is unknown.
    The environment is represented as a racetrack, and an Agent can take
    actions to evolve the environment. The possible actions that can be taken
    are (-1,-1), (-1, 0), (-1,+1), ( 0,-1), ( 0, 0), ( 0,+1), (+1,-1), (+1, 0)
    or (+1,+1). The environment is resetted when the Agent reaches the ending
    line, or exits the track.
    """

    def __init__(self, track=Track(), max_velocity=5):
        """
        Create a TrackEnv object, given a Track object. The maximum velocity
        allowed is 5.
        Input(s):   Track:          a Track object
                    max_velocity:   the maximum velocity allowed. Default is 5.
        Output(s):  None
        Example:    track_env = TrackEnv(track)
        """

        # Obtain the track description
        self.track = track
        self.width, self.height = self.track.width, self.track.height
        self.shape = self.width, self.height
        self.max_velocity = max_velocity

        # The list of possible actions an Agent can take to evolve the environment
        self.actions = np.array([(-1,-1), (-1, 0), (-1,+1), ( 0,-1), ( 0, 0), ( 0,+1), (+1,-1), (+1, 0), (+1,+1)])
        self.action_size = len(self.actions)

        self.space_size = (*self.shape, self.max_velocity+1, self.max_velocity+1)

        # The current state includes information about both
        # the position and the velocity of the car:
        # current_state = (X_pos, Y,pos, X_vel, Y_vel)
        self.current_state = (None, None, 0, 0)
        self.goal_reached = False

        # Initialize the TrackEnv to its initial state
        self.reset()

    def reset(self):
        """
        Reset the environment to a random initial state. The initial state
        is chosen randomly between the available starting positions, and the
        velocity is set to 0.
        Input(s):   None
        Output(s):  None
        Example:    track_env.reset()
        """

        # Randomly choose the initial cell between
        # the available starting positions. Also reset the velocity
        starting_index = np.random.randint(len(self.track.starting_positions))
        self.current_state = (*self.track.starting_positions[starting_index], 0, 0)

        logger.debug("Environemnt resetted to a random initial state...")

        # Set the ending state to False
        self.goal_reached = False

    def step(self, action=(0,0)):
        """
        Evolution of the environment given an action. The action is applied
        to the current state, and the new state is returned. The reward is -1
        for every step taken in the track, and 0 if the ending line is passed.
        Input(s):   action:     the action to be taken. Default is (0,0).
        Output(s):  new_state:  the new state of the environment
                    reward:     the reward obtained after the action is taken
                    done:       True if the ending line is passed, False otherwise
        Example:    new_state, reward, done = track_env.step(action)
        """

        # Assert the specified action A is one of the available actions to be taken
        assert action in self.actions, "Invalid action specified."

        # Let's suppose to does not pass the ending line, and set the reward to -1
        # (penalize for every step taken in the track)
        reward = -1
        self.goal_reached=False

        # Read the current position and velocity from the state (before
        # applying the action)
        old_position = (self.current_state[0], self.current_state[1])
        old_velocity = (self.current_state[2], self.current_state[3])

        # Calculate the new velocity after the action is taken
        new_velocity = (old_velocity[0] + action[0], old_velocity[1] + action[1])

        # Check that the velocity after the action is taken is not null,
        # and any dimension is between 0 and ´self.max_velocity´
        #if not (0 <= new_velocity[0] <= self.max_velocity and 0 <= new_velocity[1] <= self.max_velocity) or new_velocity == (0,0):
        if not (
            abs(new_velocity[0]) <= self.max_velocity
            and abs(new_velocity[1]) <= self.max_velocity
            ):
            logging.debug("New velocity can not be null, and must stay between 0 and %s in any direction. Limiting the velocity.", self.max_velocity)

            # Limiy the velocity to the maximum allowed in both directions
            new_velocity = (
                min(
                    max(new_velocity[0], -self.max_velocity),
                    self.max_velocity
                    ),
                min(
                    max(new_velocity[1], -self.max_velocity),
                    self.max_velocity
                    )
                )
        if new_velocity == (0,0):
            logging.debug("New velocity is null. Keeping the previous velocity.")
            new_velocity = old_velocity

            #self.reset()
            #return self.current_state, reward, self.goal_reached

        # Calculate the new position where the car should end
        new_position = (
            old_position[0] + new_velocity[0],
            old_position[1] + new_velocity[1]
            )

        # Now we have the new position in which the car should end
        # after the action is taken

        # Check if the car passes the ending line when the action is taken.
        # If so, reset the environment and return the ´goal_reached´ variable
        # to True
        if WooLineAlgorithm(
            np.pad(np.where(self.track.track == 3, True, False), # Ending line (padded with ´self.max_velocity´ cells
                    self.max_velocity, mode='constant',          # in order to be sure to contain
                    constant_values=False),                      # the ending position)
            (   # Starting position (in a ´self.max_velocity´-cells padded enviroment)
                old_position[0]+self.max_velocity,
                old_position[1]+self.max_velocity
            ),
            (
                new_position[0]+self.max_velocity,
                new_position[1]+self.max_velocity
            )  # Ending position (in a ´self.max_velocity´-cells padded enviroment)
            ):
            logger.debug("Win!")
            self.reset()
            self.goal_reached = True

        # The car does not pass the ending line.
        # Check if exits from the track, going outside the box or staying inside
        elif (
                not (0 <= new_position[0] < self.track.width and 0 <= new_position[1] < self.track.height)
            ) or (
                WooLineAlgorithm(
                    np.where(self.track.track == 0, True, False),
                    (old_position[0], old_position[1]),
                    (new_position[0], new_position[1])
                )
            ):
            logger.debug("Car exited the track without passing the ending line!")
            self.reset()

        # The car does not exits the track and does not pass the ending line.
        # The environment can evolve in a natural way.
        else:
            self.current_state = (*new_position, *new_velocity)

        # Return the new state, the reward and the ending condition
        return self.current_state, reward, self.goal_reached

    def render(self, plot_track=False):
        """
        Render the current state of the Environment, and optionally the track.
        Input(s):   plot_track:     if True, the track is plotted. Default is
                                    False.
        Output(s):  None
        Example:    track_env.render(plot_track=True)
        """

        fig, (ax1) = plt.subplots(1, 1, figsize=(7, 5))

        self.track._plot_track_boundary(ax1)
        
        if plot_track:
            self.track._plot_track(ax1)

        self._plot_state(ax1)

        plt.show()

    def _plot_state(self, ax):
        """
        Add the current state to the specified Pyplot axis.
        Input(s):   ax:     the axis to which the current state is added
        Output(s):  None
        Example:    track_env._plot_state(ax)
        """

        # Add the current state pointer
        ax.add_patch(
            Rectangle((self.current_state[0]-.5, self.current_state[1]-.5),
                    width=1,
                    height=1,
                    color='orange',
                    fill=True)
        )

        ax.annotate("",
                    xy=(
                        self.current_state[0]+self.current_state[2],
                        self.current_state[1]+self.current_state[3]),
                    xytext=(self.current_state[0], self.current_state[1]),
                    arrowprops=dict(arrowstyle='->'),
                    annotation_clip=False
                    )

        return ax


class TDControl():
    """
    This class implements a Temporal Difference Control object.
    """

    def __init__(self, environment, lambda_=0, gamma=1, lr_v=0.01, epsilon=0.01):
        """
        Create a Temporal Difference Control object.
        Input(s):   environment:    a TrackEnv object
                    lambda_:        the lambda parameter for the eligibility
                                    traces. Default is 0.
                    gamma_          the discount factor. Default is 1.
                    lr_v:           the learning rate. Default is 0.01.
                    epsilon:        the epsilon parameter for the epsilon-greedy
        Output(s):  None
        Example:    control = TDControl(environment=TrackEnv(), lambda_=0, gamma=1, lr_v=0.01, epsilon=0.01)
        """

        self.name = "TDControl"

        # Save the space size, the actions and the action size of the environment
        # the control object is interacting with
        self.environment = environment
        self.space_size = self.environment.space_size
        self.actions = self.environment.actions
        self.action_size = self.environment.action_size

        # Lambda value for TD(lambda)
        assert 0 <= lambda_ <= 1, "Lambda must be between 0 and 1."
        self.lambda_ = lambda_

        # Discount factor
        assert 0 <= gamma <= 1, "Discount factor must be between 0 and 1."
        self.gamma = gamma

        # Learning rate
        assert 0 <= lr_v <= 1, "Learning rate must be between 0 and 1."
        self.lr_v = lr_v

        # Epsilon for epsilon-greedy policy
        assert 0 <= epsilon <= 1, "Epsilon must be between 0 and 1."
        self.epsilon = epsilon

        # Array of Qvalues
        self.Qvalues = np.zeros( (*self.space_size, self.action_size) )

        # Initialize the number of epochs, the performances
        # and trajectories arrays
        self.epochs = 0
        self.performances = []
        self.trajectories = []

    def control(self, n_episodes=1_000):
        """
        Run the control loop for a specified number of episodes.
        The Qvalues and the performances are saved on filesystem.
        Input(s):   n_episodes:     the number of episodes to run. Default is
                                    1_000.
        Output(s):  None
        Example:    control.learn(n_episodes=1_000)
        """

        for _ in tqdm(range(self.epochs, self.epochs+n_episodes)):

            self.performances.append(0)
            self.trajectories.append([])

            done = False

            self.environment.reset()

            current_state = self.environment.current_state
            current_action_no = self.get_action_epsilon_greedy(current_state)
            current_action = self.actions[current_action_no]

            while not done:

                # Evolve one step
                new_state, reward, done = self.environment.step(current_action)

                self.trajectories[_].append((current_state, current_action))

                # Keeps track of performance for each episode
                self.performances[_] += reward

                # Choose new action index
                new_action_no = self.get_action_epsilon_greedy(new_state)
                # (Corresponding action to index)
                current_action = self.actions[new_action_no]

                # Single step update
                self.single_step_update(
                    current_state=current_state,
                    current_action_no=current_action_no,
                    reward=reward,
                    new_state=new_state,
                    new_action_no=new_action_no,
                    done=done
                    )

                current_action_no = new_action_no
                current_state = new_state

            # Increment the number of episodes run
            self.epochs += 1

        # Save the Qvalues and the performances on filesystem
        self.save()

        # Plot the learning curve
        #self.plot_performance()

        return

    def get_action_epsilon_greedy(self, current_state):
        """
        Return the index of the action to be taken, given the current state
        and the epsilon-greedy policy.
        Input(s):   current_state:  the current state of the environment
        Output(s):  action_no:      the index of the action to be taken
        Example:    action_no = td_control.get_action_epsilon_greedy(current_state)
        """

        rand = np.random.rand()

        if rand < self.epsilon:
            logger.debug("Action choosen randomly!")
            # probability is uniform for all actions!
            prob_actions = np.ones(self.action_size) / self.action_size

        else:

            logger.debug("Action choosen NOT randomly!")
            # I find the best Qvalue
            best_value = np.max(self.Qvalues[ (*current_state,) ])

            # There could be actions with equal value!
            best_actions = self.Qvalues[ (*current_state,) ] == best_value

            # best_actions is an array of booleans, where:
            # *True* if the value is equal to the best (possibly ties)
            # *False* if the action is suboptimal
            prob_actions = best_actions / np.sum(best_actions)

        # take one action from the array of actions with the probabilities as defined above.
        return np.random.choice(self.action_size, p=prob_actions)

    def single_step_update(self, current_state, current_action_no,
        reward, new_state, new_action_no, done=False):
        """
        Abstract method for the single step update. This method is only
        implemented in the subclasses (i.e., SARSA, QLearning and ExpectedSARSA).
        Input(s):   current_state:      the current state of the environment
                    current_action_no:  the index of the action taken
                    reward:             the reward obtained after the action is taken
                    new_state:          the new state of the environment
                    new_action_no:      the index of the new action taken
                    done:               True if the ending line is passed, False otherwise
        Output(s):  None
        Example:    td_control.single_step_update(current_state, current_action_no,
                                reward, new_state, new_action_no, done=False)
        """
        raise NotImplementedError('The method must be implemented in the subclasses!')

    def plot_performance(self):
        """
        Plot the learning curve of the Control.
        Input(s):   None
        Output(s):  None
        Example:    control.plot_performance()
        """

        # Specify the plot size
        plt.figure(figsize=(14, 6))

        plt.suptitle('Rewards along epochs', fontsize = 12)
        plt.plot(self.performances, label=self.name)
        plt.yscale('symlog')
        plt.xlabel("Epochs")
        plt.ylabel("Rewards")
        plt.legend()
        plt.grid(True, linestyle='dotted', linewidth=.5)
        plt.show()

    def render(self, epoch_no=-1, plot_track=False):
        """
        Render the trajectory for the specified epoch, and optionally the
        track.
        Input(s):   epoch_no:       the epoch to be rendered. Default is -1
                                    (last epoch).
                    plot_track:     if True, the track is also plotted. Default
                                    is False.
        Output(s):  None
        Example:    control.render(epoch_no=1_000, plot_track=True)
        """

        fig, (ax1) = plt.subplots(1, 1, figsize=(7, 5))

        self.environment.track._plot_track_boundary(ax1)

        if plot_track:
            self.environment.track._plot_track(ax1)

        self._plot_trajectory(ax1, trajectory_no=epoch_no)

        plt.show()

    def _plot_trajectory(self, ax, trajectory_no=-1):
        """
        Add the trajectory for the specified epoch to the specified Pyplot axis.
        Input(s):   ax:             the axis to which the trajectory is added
                    trajectory_no:   the epoch to be rendered. Default is -1
                                    (last epoch).
        Output(s):  None
        Example:    control._plot_trajectory(ax, trajectory_no=1_000)
        """

        for state, action in self.trajectories[trajectory_no]:

            current_position = (state[0], state[1])
            current_velocity = (state[2], state[3])
            new_velocity = (
                current_velocity[0] + action[0],
                current_velocity[1] + action[1]
                )

            ax.add_patch(
                Rectangle((current_position[0]-.5, current_position[1]-.5),
                    width=1,
                    height=1,
                    color='orange',
                    fill=True)
            )

            ax.annotate("",
                    xy=(
                        current_position[0]+current_velocity[0],
                        current_position[1]+current_velocity[1]
                        ),
                    xytext=(current_position[0], current_position[1]),
                    arrowprops=dict(
                        arrowstyle='->', linestyle='dashed', alpha=0.5
                        ),
                    annotation_clip=False
                    )
            ax.annotate("",
                    xy=(
                        current_position[0]+current_velocity[0]+action[0],
                        current_position[1]+current_velocity[1]+action[1]
                        ),
                    xytext=(
                        current_position[0]+current_velocity[0],
                        current_position[1]+current_velocity[1]
                        ),
                    arrowprops=dict(
                        arrowstyle='->', color='red',
                        linestyle='dashed', alpha=0.5
                        ),
                    annotation_clip=False
                    )
            ax.annotate("",
                    xy=(
                        current_position[0]+new_velocity[0],
                        current_position[1]+new_velocity[1]
                        ),
                    xytext=(current_position[0], current_position[1]),
                    arrowprops=dict(arrowstyle='->'),
                    annotation_clip=False
                    )

        return

    def save(self, file=None):
        """
        Save the Control object on filesystem. The Qvalues, the performances
        and the trajectories are saved.
        Input(s):   file:   the file where to save the Control object. Default
                            is "./data/{self.name}_Control.pkl".
        Output(s):  None
        Example:    control.save(file='./data/{self.name}_Control.pkl')
        """

        if not file:
            file = f"./data/{self.name}_Control.pkl"
            logger.debug("Saving to default file location ""%s"".", file)

        with open(file, 'wb') as output_file:
            pickle.dump(self, output_file, pickle.HIGHEST_PROTOCOL)

        logger.debug("%s Control files saved into ""%s"".", self.name, file)

    def load(self, file=None):
        """
        Load from filesystem the current Qvalues, the performances and the
        trajectories.
        Input(s):   file:   the file from where to load the Control object.
                            Default is "./data/{self.name}_Control.pkl".
        Output(s):  None
        Example:    control.load(file='./data/{self.name}_Control.pkl')
        """

        if not file:
            file = f"./data/{self.name}_Control.pkl"
            logger.debug("Loading from default file ""%s"".", file)

        with open(file, 'rb') as input_file:
            obj_dict = pickle.load(input_file).__dict__
            self.__dict__.update(obj_dict)

        logger.info("%s Control restored from ""%s"" file.", self.name, file)

        return self


class SARSA(TDControl):
    """
    This class implements a SARSA control object.
    """

    def __init__(self, environment, lambda_=0, gamma=1, lr_v=0.01, epsilon=0.01):
        """
        Create a SARSA object.
        Input(s):   environment:    a TrackEnv object
                    lambda_:        the lambda parameter for the eligibility
                                    traces. Default is 0.
                    gamma_          the discount factor. Default is 1.
                    lr_v:           the learning rate. Default is 0.01.
                    epsilon:        the epsilon parameter for the epsilon-greedy
                                    policy. Default is 0.01.
        Output(s):  None
        Example:    sarsa = SARSA(environment=TrackEnv(), lambda_=0, gamma=1, lr_v=0.01, epsilon=0.01)
        """

        super().__init__(environment, lambda_, gamma, lr_v, epsilon)
        self.name = "SARSA"

    def single_step_update(self, current_state, current_action_no,
        reward, new_state, new_action_no, done=False):
        """
        Update the Qvalues using the current state, the action taken, the
        reward obtained, the new state and the new action taken.
        Input(s):   current_state:      the current state of the environment
                    current_action_no:  the index of the action taken
                    reward:             the reward obtained after the action is
                                        taken
                    new_state:          the new state of the environment
                    new_action_no:      the index of the new action taken
                    done:               True if the ending line is passed,
                                        False otherwise
        Output(s):  None
        Example:    sarsa.single_step_update(current_state, current_action_no,
                                reward, new_state, new_action_no, done=False)
        """

        # Calculate the TD erro using the SARSA approach
        if done:
            td_error = reward + 0 - self.Qvalues[ (*current_state, current_action_no) ]
        else:
            td_error = (reward +
                      self.gamma * self.Qvalues[ (*new_state, new_action_no) ]
                                 - self.Qvalues[ (*current_state, current_action_no) ])

        # Update the Qvalues using the TD error
        self.Qvalues[ (*current_state, current_action_no) ] += self.lr_v * td_error


class QLearning(TDControl):

    def __init__(self, environment, lambda_=0, gamma=1, lr_v=0.01, epsilon=0.01):
        """
        Create a QLearning object.
        Input(s):   environment:    a TrackEnv object
                    lambda_:        the lambda parameter for the eligibility
                                    traces. Default is 0.
                    gamma_          the discount factor. Default is 1.
                    lr_v:           the learning rate. Default is 0.01.
                    epsilon:        the epsilon parameter for the epsilon-greedy
                                    policy. Default is 0.01.
        Output(s):  None
        Example:    q_learning = QLearning(environment=TrackEnv(), lambda_=0, gamma=1, lr_v=0.01, epsilon=0.01)
        """
        super().__init__(environment, lambda_, gamma, lr_v, epsilon)
        self.name = "QLearning"

    def single_step_update(self, current_state, current_action_no,
        reward, new_state, done=False, **kwargs):
        """
        Update the Qvalues using the current state, the action taken, the
        reward obtained, the new state and the ending condition.
        Input(s):   current_state:      the current state of the environment
                    current_action_no:  the index of the action taken
                    reward:             the reward obtained after the action is taken
                    new_state:          the new state of the environment
                    done:               True if the ending line is passed, False otherwise
                    **kwargs:           additional arguments (not used, but useful
                                        for compatibility with SARSA control method)
        Output(s):  None
        Example:    q_learning.single_step_update(current_state, current_action_no,
                                reward, new_state, done=False)
        """

        # Calculate the TD error using the Q-Learning approach
        if done:
            td_error = reward + 0 - self.Qvalues[ (*current_state, current_action_no) ]
        else:
            td_error = (reward +
                      self.gamma * np.max(self.Qvalues[ (*new_state,) ])
                                 - self.Qvalues[ (*current_state, current_action_no) ])

        # Update the Qvalues using the TD error
        self.Qvalues[ (*current_state, current_action_no) ] += self.lr_v * td_error


class ExpectedSARSA(TDControl):
    """
    This class implements an Expected SARSA Control object.
    """

    def __init__(self, environment, lambda_=0, gamma=1, lr_v=0.01, epsilon=0.01):
        """
        Create an ExpectedSARSA object. The Expected SARSA algorithm is an
        off-policy TD control algorithm.
        Input(s):   environment:    a TrackEnv object
                    lambda_:        the lambda parameter for the eligibility
                                    traces. Default is 0.
                    gamma_          the discount factor. Default is 1.
                    lr_v:           the learning rate. Default is 0.01.
                    epsilon:        the epsilon parameter for the epsilon-greedy
                                    policy. Default is 0.01.
        Output(s):  None
        Example:    expected_sarsa = ExpectedSARSA(environment=TrackEnv(), lambda_=0, gamma=1, lr_v=0.01, epsilon=0.01)
        """

        super().__init__(environment, lambda_, gamma, lr_v, epsilon)
        self.name = "ExpectedSARSA"

    def single_step_update(self, current_state, current_action_no,
        reward, new_state, done=False, **kwargs):
        """
        Update the Qvalues using the current state, the action taken, the
        reward obtained, the new state and the ending condition.
        Input(s):   current_state:      the current state of the environment
                    current_action_no:  the index of the action taken
                    reward:             the reward obtained after the action is taken
                    new_state:          the new state of the environment
                    done:               True if the ending line is passed, False otherwise
        Output(s):  None
        Example:    expected_sarsa.single_step_update(current_state, current_action_no,
                                reward, new_state, done=False)
        """

        # Calculate the TD error using the Expected SARSA approach
        if done:
            td_error = reward + 0 - self.Qvalues[ (*current_state, current_action_no) ]
        else:
            td_error = (reward +
                      self.gamma * np.dot(
                        self.Qvalues[ (*new_state,)],
                        self.policy(new_state)
                        )
                    - self.Qvalues[ (*current_state, current_action_no) ])

        # Update the Qvalues using the TD error
        self.Qvalues[ (*current_state, current_action_no) ] += self.lr_v * td_error

    def greedy_policy(self):
        """
        Calculates the greedy policy given current Qvalues.
        Input(s):   None
        Output(s):  policy:     the greedy policy
        Example:    policy = expected_sarsa.greedy_policy()
        """
        return np.argmax(self.Qvalues, axis = -1)
        
    def policy(self, new_state):
        """
        Probabilities from an epsilon-greedy policy wrt the current Q(s,a).
        Input(s):   new_state:  the new state of the environment
        Output(s):  policy:     the epsilon-greedy policy
        Example:    policy = expected_sarsa.policy(new_state)
        """

        # Uniform (epsilon) probability for all actions...
        policy = np.ones(self.action_size) / self.action_size * self.epsilon
        # ... plus 1-epsilon probabilities for best actions:
        # First I find the best values
        best_value = np.max(self.Qvalues[ (*new_state,) ])
        # There could be actions with equal value!
        # This mask is 1 if the value is equal to the best (tie)
        # or 0 if the action is suboptimal
        best_actions = (self.Qvalues[ (*new_state,) ] == best_value)
        policy += best_actions / np.sum(best_actions) * (1 - self.epsilon)
        return policy
