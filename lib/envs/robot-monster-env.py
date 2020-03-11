import numpy as np


# Interface
class Environment(object):

    def reset(self):
        raise NotImplementedError('Inheriting classes must override reset.')

    def actions(self):
        raise NotImplementedError('Inheriting classes must override actions.')

    def step(self, action):
        raise NotImplementedError('Inheriting classes must override step')


class ActionSpace(object):

    def __init__(self, actions):
        self.actions = actions
        self.n = len(actions)


# Robot-Monster-Game Environment

class RobotMonsterEnv(Environment):
    """Define a robot-monster-game environment"""
    """actions: 0 - north, 1 - east, 2 - west, 3 - south"""

    def __init__(self):
        super(RobotMonsterEnv, self).__init__()

        # define state and action space
        s_pos = [i for i in range(25)]
        s_d = [False, True]
        s_prize = [-1, 0, 4, 20, 24]  # P=0 if there is no prize, P=i if there is a prize at position Pos_i
        self.S = [(i, j, k) for i in s_pos for j in s_d for k in s_prize]
        self.action_space = ActionSpace(range(4))

        # define transitions
        self.Pos = {}
        self.Pos[0] = [1, 5]
        self.Pos[1] = [0, 2, 6]
        self.Pos[2] = [1, 3, 7]
        self.Pos[3] = [2, 4, 8]
        self.Pos[4] = [3, 9]
        self.Pos[5] = [0, 6, 10]
        self.Pos[6] = [1, 5, 7, 11]
        self.Pos[7] = [2, 6, 8, 12]
        self.Pos[8] = [3, 7, 9, 13]
        self.Pos[9] = [4, 8, 14]
        self.Pos[10] = [5, 11, 15]
        self.Pos[11] = [6, 10, 12, 16]
        self.Pos[12] = [7, 11, 13, 17]
        self.Pos[13] = [8, 12, 14, 18]
        self.Pos[14] = [9, 13, 19]
        self.Pos[15] = [10, 20]
        self.Pos[16] = [11, 17, 21]
        self.Pos[17] = [12, 16, 18, 22]
        self.Pos[18] = [13, 17, 19, 23]
        self.Pos[19] = [14, 18, 24]
        self.Pos[20] = [15]
        self.Pos[21] = [16]
        self.Pos[22] = [17, 23]
        self.Pos[23] = [18, 22, 24]
        self.Pos[24] = [19, 23]

        self.max_trajectory_length = 50
        self.tolerance = 0.1
        self._rendered_maze = self._render_maze()

    def step(self, action):
        s_prev = self.s
        self.s = self.single_step(self.s, action)
        reward = self.single_reward(self.s, s_prev)
        self.nstep += 1
        self.is_reset = False

        if (reward < -1. * self.tolerance or reward > self.tolerance) or self.nstep == self.max_trajectory_length:
            self.reset()

        return self._convert_state(self.s), reward, self.is_reset, ''

    def single_step(self, s, a):
        """actions: 0 - north, 1 - east, 2 - west, 3 - south"""
        if a < 0 or a > 3:
            raise ValueError('Unknown action', a)
        if a == 0 and (s - 4 in self.Pos[s]):
            s += 5
        elif a == 1 and (s + 1 in self.Pos[s]):
            s += 1
        elif a == 2 and (s - 1 in self.Pos[s]):
            s -= 1
        elif a == 3 and (s + 4 in self.Pos[s]):
            s -= 5
        return s

    def single_reward(self, s, s_prev):
        r = 0
        if s[2] != -1:
            r += 10
        if s[0] == monster_pos:
            if s[1] is True:
                r -= 10
        if s[0] == s_prev[0]:
            r -= 1
        return r

    def reset(self):
        self.nstep = 0
        self.s = np.random.randint(0, 25)
        self.is_reset = True
        return self._convert_state(self.s)

    def _convert_state(self, s):
        converted = np.zeros(len(self.S), dtype=np.float32)
        converted[s] = 1
        return converted

    def _get_render_coords(self, s):
        return (int(s / 4) * 4, (s % 4) * 4)

    def _render_maze(self):
        # draw background and grid lines
        maze = np.zeros((17, 17))
        for x in range(0, 17, 4):
            maze[x, :] = 0.5
        for y in range(0, 17, 4):
            maze[:, y] = 0.5

        # draw reward and transitions
        for s in range(16):
            if self.R[s] != 0:
                x, y = self._get_render_coords(s)
                maze[x + 1:x + 4, y + 1:y + 4] = self.R[s]
            if self.single_step(s, 0) == s:
                x, y = self._get_render_coords(s)
                maze[x, y:y + 5] = -1
            if self.single_step(s, 1) == s:
                x, y = self._get_render_coords(s)
                maze[x:x + 5, y + 4] = -1
            if self.single_step(s, 2) == s:
                x, y = self._get_render_coords(s)
                maze[x:x + 5, y] = -1
            if self.single_step(s, 3) == s:
                x, y = self._get_render_coords(s)
                maze[x + 4, y:y + 4] = -1
        return maze

    def render(self, mode='rgb_array'):
        assert mode == 'rgb_array', 'Unknown mode: %s' % mode
        img = np.array(self._rendered_maze, copy=True)

        # draw current agent location
        x, y = self._get_render_coords(self.s)
        img[x + 1:x + 4, y + 1:y + 4] = 2.0
        return img
