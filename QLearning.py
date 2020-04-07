import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time
# from ImageProcessing import average_red

style.use("ggplot")

SIZE = 10
HM_EPISODES = 25000
MOVE_PENALTY = 1
TARGET_REWARD = 25
ROCK_PENALTY = 300
epsilon = 0.7
epsilon_decay = 0.6998
SHOW_EVERY = 3000

# start_q_table = "C:/Users/nsree/PycharmProjects/EDL/qtable - 1580149065.pickle"
start_q_table = None
LEARNING_RATE = 0.7
DISCOUNT = 0.95

AGENT_N = 1
TARGET_N = 2
ROCK_N = 3

d = {1: (255, 175, 0),
     2: (0, 255, 0),
     3: (0, 0, 255)}


class HIAD:
    def __init__(self):
        self.x = np.random.randint(0, SIZE)
        self.y = np.random.randint(0, SIZE)

    def __sub__(self, other):
        return (self.x-other.x, self.y-other.y)

    def action(self, choice):
        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=-1)
        elif choice == 2:
            self.move(x=-1, y=1)
        elif choice == 3:
            self.move(x=1, y=-1)
        elif choice == 4:
            self.move(x=1, y=0)
        elif choice == 5:
            self.move(x=0, y=1)

    def move(self, x=False, y=False):
        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x
        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y
        if self.x < 0:
            self.x = 0
        elif self.x > SIZE-1:
            self.x = SIZE - 1
        if self.y < 0:
            self.y = 0
        elif self.y > SIZE-1:
            self.y = SIZE - 1


if start_q_table is None:
    q_table = {}
    for x1 in range(-SIZE+1, SIZE):
        for y1 in range(-SIZE+1, SIZE):
            for x2 in range(-SIZE + 1, SIZE):
                for y2 in range(-SIZE + 1, SIZE):
                    q_table[((x1, y1), (x2, y2))] = [np.random.uniform(-5, 0) for i in range(4)]

else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)
print(q_table)

episode_rewards = []

for episode in range(HM_EPISODES):
    agent = HIAD()
    target = HIAD()
    rock = HIAD()
    if episode % SHOW_EVERY == 0:
        print(f"on #{episode}, epsilon is {epsilon}")
        print(f"{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False

    episode_reward = 0
    for i in range(200):
        obs = (agent - target, agent - rock)
        # print(obs)
        if np.random.random() > epsilon:
            action = np.argmax(q_table[obs])
        else:
            action = np.random.randint(0, 4)
        agent.action(action)

        if agent.x == target.x and agent.y == target.y:
            reward = TARGET_REWARD
        elif agent.x == rock.x and agent.y == rock.y:
            reward = -ROCK_PENALTY
        else:
            reward = -MOVE_PENALTY

        new_obs = (agent - target, agent - rock)
        max_future_q = np.max(q_table[new_obs])
        current_q = q_table[obs][action]

        if reward == TARGET_REWARD:
            new_q = TARGET_REWARD
        else:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        q_table[obs][action] = new_q

        if show:
            env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
            env[target.x][target.y] = d[TARGET_N]
            env[agent.x][agent.y] = d[AGENT_N]
            env[rock.x][rock.y] = d[ROCK_N]
            img = Image.fromarray(env, 'RGB')
            # img2 = cv2.imread('C:/Users/nsree/Desktop/fig.jpg')
            img = img.resize((300, 300))
            # img2 = img2.resize((300, 300))
            # img3 = cv2.add(img, img2)
            # img3 = img3.resize((3000, 3000))
            cv2.imshow("image", np.array(img))
            if reward == TARGET_REWARD or reward == ROCK_PENALTY:
                if cv2.waitKey(500) & 0xFF == ord('q'):
                    break
            else:
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

        episode_reward += reward
        if reward == TARGET_REWARD or reward == -ROCK_PENALTY:
            break

    episode_rewards.append(episode_reward)
    epsilon *= epsilon_decay

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,))/SHOW_EVERY, mode='valid')

plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"Reward every {SHOW_EVERY} episodes")
plt.xlabel("episode number")
plt.show()

with open(f"qtable - {int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table, f)
