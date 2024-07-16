import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt

class ContinuousMazeEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, maze_size=(5.0, 5.0), max_distance=5.0, obstacles=None):
        super(ContinuousMazeEnv, self).__init__()

        self.maze_size = np.array(maze_size, dtype=np.float32)
        self.max_distance = max_distance
        self.agent_position = np.array([0.0, 0.0], dtype=np.float32)
        self.goal_position = np.array([maze_size[0] - 1, maze_size[1] - 1], dtype=np.float32)
        
        # 定义障碍物
        self.obstacles = obstacles if obstacles is not None else []

        # 动作空间：离散的方向(0-4)，连续的前进距离
        self.action_space = spaces.Tuple((
            spaces.Discrete(5),  # 0: 呆在原地, 1: 上, 2: 下, 3: 左, 4: 右
            spaces.Box(low=0, high=max_distance, shape=(1,), dtype=np.float32)
        ))

        # 状态空间：当前代理的位置
        self.observation_space = spaces.Box(
            low=np.array([0, 0], dtype=np.float32), 
            high=self.maze_size,
            shape=(2,), 
            dtype=np.float32
        )

    def reset(self):
        self.agent_position = np.array([0.0, 0.0], dtype=np.float32)
        return self.agent_position

    def step(self, action):
        direction, distance = action
        distance = distance[0]

        new_position = self.agent_position.copy()
        if direction == 1:  # 上
            new_position[1] = min(self.maze_size[1], self.agent_position[1] + distance)
        elif direction == 2:  # 下
            new_position[1] = max(0, self.agent_position[1] - distance)
        elif direction == 3:  # 左
            new_position[0] = max(0, self.agent_position[0] - distance)
        elif direction == 4:  # 右
            new_position[0] = min(self.maze_size[0], self.agent_position[0] + distance)

        # 检查新位置是否碰到障碍物
        if self._check_collision(new_position):
            reward = -10  # 碰到障碍物惩罚
            done = False
        else:
            self.agent_position = new_position
            done = np.array_equal(self.agent_position, self.goal_position)
            reward = 1 if done else -0.1

        return self.agent_position, reward, done, {}

    def _check_collision(self, position):
        for obstacle in self.obstacles:
            if obstacle.collide(position):
                return True
        return False

    def render(self, mode='human'):
        plt.figure(figsize=(6, 6))
        plt.xlim(0, self.maze_size[0])
        plt.ylim(0, self.maze_size[1])
        
        # 绘制代理的位置
        plt.plot(self.agent_position[0], self.agent_position[1], 'ro')  # 红点表示代理
        
        # 绘制目标的位置
        plt.plot(self.goal_position[0], self.goal_position[1], 'go')  # 绿点表示目标
        
        # 绘制障碍物的位置
        for obstacle in self.obstacles:
            plt.gca().add_patch(plt.Rectangle(
                (obstacle.x, obstacle.y), obstacle.width, obstacle.height, color='gray'
            ))
        
        plt.grid(True)
        plt.show()

    def close(self):
        pass

class Obstacle:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def collide(self, position):
        return (self.x <= position[0] <= self.x + self.width) and (self.y <= position[1] <= self.y + self.height)

# 使用环境
obstacles = [
    Obstacle(1.5, 1.5, 1.0, 1.0),
    Obstacle(3.0, 3.0, 1.5, 1.5)
]

env = ContinuousMazeEnv(obstacles=obstacles)
obs = env.reset()

for _ in range(5):
    action = env.action_space.sample()
    print(f"Action: {action}")
    obs, reward, done, info = env.step(action)
    env.render()
    print(f"Position: {obs}, Reward: {reward}, Done: {done}")
    if done:
        break
env.close()
