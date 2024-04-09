from numpy import multiply, zeros, add, random, count_nonzero
from numba import njit


class MineSweeper():
    def __init__(self, width, height, num_bombs):
        self.maze_width = width
        self.maze_height = height
        self.num_bombs = num_bombs
        self.box_count = self.maze_width * self.maze_height
        self.uncovered_count = 0
        self.reset()

    def reset(self):
        self.grid = zeros((self.maze_width, self.maze_height), dtype=int)
        self.fog = zeros((self.maze_width, self.maze_height), dtype=int)
        self.state = zeros((self.maze_width, self.maze_height), dtype=int)
        self.bomb_locs = random.choice(range(self.box_count), self.num_bombs, replace=False)
        self.deploy_bombs()
        self.hint_maker()
        self.update_state()
        self.uncovered_count = 0

    def update_state(self):
        self.state = multiply(self.grid, self.fog)
        self.state = add(self.state, (self.fog - 1))

    def deploy_bombs(self):  # plant_bombs to deploy_bombs
        reordered_bomb_locs = []
        maze_width = self.maze_width
        for bomb_loc in self.bomb_locs:
            row = int(bomb_loc / maze_width)
            col = int(bomb_loc % maze_width)
            self.grid[row][col] = -1
            reordered_bomb_locs.append((row, col))
        self.bomb_locs = reordered_bomb_locs

    def hint_maker(self):
        maze_height = self.maze_height
        maze_width = self.maze_width
        for r, c in self.bomb_locs:
            for i in range(r - 1, r + 2):
                for j in range(c - 1, c + 2):
                    if i > -1 and j > -1 and i < maze_height and j < maze_width and self.grid[i][j] != -1:
                        self.grid[i][j] += 1

    def choose(self, i, j):
        if (self.grid[i][j] == 0):
            unfog_zeros(self.grid, self.fog, i, j)
            self.uncovered_count = count_nonzero(self.fog)
            self.update_state()
            if (self.uncovered_count == self.box_count - self.num_bombs):
                return self.state, True, 1
            return self.state, False, 0.5
        elif (self.grid[i][j] > 0):
            self.fog[i][j] = 1
            self.uncovered_count = count_nonzero(self.fog)
            self.update_state()
            if (self.uncovered_count == self.box_count - self.num_bombs):
                return self.state, True, 1
            return self.state, False, 0.5
        else:
            return self.state, True, -1


@njit(fastmath=True)
def unfog_zeros(grid, fog, i, j):
    h, w = grid.shape
    queue = []
    queue.append((i, j))
    while (len(queue) > 0):
        i, j = queue.pop()
        for r in range(i - 1, i + 2):
            for c in range(j - 1, j + 2):
                if (r >= 0 and r < h and c >= 0 and c < w):
                    if (grid[r][c] == 0 and fog[r][c] == 0):
                        queue.append((r, c))
                    fog[r][c] = 1
