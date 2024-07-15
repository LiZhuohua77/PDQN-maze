from multilane import MultiLaneVehicleEnvironment
from agents.pdqn import PDQNAgent
from gymnasium import spaces
import argparse
import torch
import numpy as np
import datetime
import os
import time
from torch.utils.tensorboard import SummaryWriter

SEQUENCE = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def pad_action(act, all_action_parameters):
    # 确定 action_parameter_size，用于分割 all_action_parameters
    action_parameter_size = 1  # 根据你的需求调整这个值
    
    # 分割 all_action_parameters
    split_action_params = np.split(all_action_parameters, len(all_action_parameters) // action_parameter_size)
    
    # 将 act 和分割后的参数组合成一个元组
    padded_action = (act, *split_action_params)
    
    return padded_action

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default=True, type=bool, help="Train mode or not")
    parser.add_argument('--seed', default=6, type=int)
    parser.add_argument('--save_dir', default=r'D:\Code\PDQN_test-master\results\ramp', type=str)
    parser.add_argument('--max_steps', default=5000, type=int)
    parser.add_argument('--train_eps', default=10000, type=int)
    parser.add_argument('--eval_eps', default=1000, type=int)
    parser.add_argument('--device', default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--epsilon_initial', default=0.95, type=float)
    parser.add_argument('--epsilon_steps', default=80000, type=int)
    parser.add_argument('--epsilon_final', default=0.02, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--actor_lr', default=1e-3, type=float)
    parser.add_argument('--param_net_lr', default=1e-4, type=float)
    parser.add_argument('--gamma', default=0.90, type=float)
    parser.add_argument('--layers_actor', default=[256, 128, 64])
    parser.add_argument('--layers_param', default=[256, 128, 64])

    config = parser.parse_args()
    return config

def evaluate(env, agent, episodes=1000, writer=None):
    returns = []
    timesteps = []
    goals = []

    for i_eps in range(episodes):
        state,_ = env.reset()
        terminated = False
        truncated = False
        n_steps = 0
        total_reward = 0.
        info = {'status': "NOT_SET"}
        while not terminated and not truncated:
            n_steps += 1
            state = np.array(state, dtype=np.float32, copy=False)
            act, act_param, all_action_parameters = agent.act(state)
            action = pad_action(act, act_param)
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
        # print(info['status'])
        goal = info['status'] == 'GOAL'
        timesteps.append(n_steps)
        returns.append(total_reward)
        goals.append(goal)
        writer.add_scalar('rewards', returns[-1], i_eps)
        writer.add_scalar('timesteps', timesteps[-1], i_eps)

    writer.close()
    return np.column_stack((returns, timesteps, goals))


def train(seed, episodes, batch_size, gamma, inverting_gradients, initial_memory_threshold, replay_memory_size,
        epsilon_steps, learning_rate_actor, learning_rate_actor_param, title, epsilon_start, epsilon_final, clip_grad, beta,
        scale_actions, evaluation_episodes, update_ratio, save_freq, save_dir, layers):
    if save_freq > 0 and save_dir:
        save_dir = os.path.join(save_dir, title + "{}".format(str(seed)))
        os.makedirs(save_dir, exist_ok=True)

    env = MultiLaneVehicleEnvironment(num_vehicles=5)
    dir = os.path.join(save_dir, title)
    print("Saving to:", save_dir)
    print("Directory:", dir)

    np.random.seed(seed)


    agent = PDQNAgent(observation_space=env.observation_space, action_space=env.action_space,
                        epsilon_initial=cfg.epsilon_initial, epsilon_steps=cfg.epsilon_steps,epsilon_final=cfg.epsilon_final,
                        batch_size=cfg.batch_size, device=cfg.device, gamma=cfg.gamma,
                        actor_kwargs={"hidden_layers": cfg.layers_actor},
                        actor_param_kwargs={"hidden_layers": cfg.layers_param},
                        )
    #print(agent)
    network_trainable_parameters = sum(p.numel() for p in agent.actor.parameters() if p.requires_grad)
    network_trainable_parameters += sum(p.numel() for p in agent.actor_param.parameters() if p.requires_grad)
    print("Total Trainable Network Parameters: %d" % network_trainable_parameters)
    max_steps = 5000
    returns = []
    timesteps = []
    goals = []
    moving_avg_rewards = []
    start_time_train = time.time()
    log_dir = os.path.split(os.path.abspath(__file__))[0] + "/logs/train/" + title + "__" + str(seed) + " " + SEQUENCE
    print("Logging to:", log_dir)
    writer = SummaryWriter(log_dir)

    for i_eps in range(episodes):
        if save_freq > 0 and save_dir and i_eps % save_freq == 0:
            agent.save_models(os.path.join(save_dir, str(i_eps)))
        info = {'status': "NOT_SET"}
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32, copy=False)

        act, act_param, all_action_parameters = agent.act(state)
        action = pad_action(act, act_param)
        episode_reward = 0.
        transitions = []
        for i_step in range(max_steps):
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = np.array(next_state, dtype=np.float32, copy=False)

            next_act, next_act_param, next_all_action_parameters = agent.act(next_state)
            next_action = pad_action(next_act, next_act_param)
            transitions.append([state, np.concatenate(([act], all_action_parameters.data)).ravel(), reward,
                                next_state, np.concatenate(([next_act],
                                                next_all_action_parameters.data)).ravel(), terminated, truncated])

            act, act_param, all_action_parameters = next_act, next_act_param, next_all_action_parameters
            action = next_action
            state = next_state
            episode_reward += reward

            if terminated or truncated:
                break
        agent.end_episode()

        # calculate n-step returns
        n_step_returns = compute_n_step_returns(transitions, gamma)
        for t, nsr in zip(transitions, n_step_returns):
            t.append(nsr)
            agent.replay_memory.append(state=t[0], action_with_param=t[1], reward=t[2], next_state=t[3],
                                       done=t[5], n_step_return=nsr)

        n_updates = int(update_ratio * i_step)
        for _ in range(n_updates):
            agent._optimize_td_loss()

        if i_eps % 2 == 0:
            agent.actor_target.load_state_dict(agent.actor.state_dict())
            agent.actor_param_target.load_state_dict(agent.actor_param.state_dict())

        returns.append(episode_reward)
        timesteps.append(i_step)
        goals.append(info['status'] == 'GOAL')

        if i_eps == 0:
            moving_avg_rewards.append(episode_reward)
        else:
            moving_avg_rewards.append(episode_reward * 0.1 + moving_avg_rewards[-1] * 0.9)

        if i_eps % 100 == 0 and i_eps > 0:
            print('Episode: ', i_eps, 'R: ', moving_avg_rewards[-1], 'n_steps: ',
                  np.array(timesteps[-100]).mean(), 'epsilon: ', agent.epsilon)

        writer.add_scalars('rewards', {'raw': returns[-1], 'moving_average': moving_avg_rewards[-1]}, i_eps)
        writer.add_scalar('step_of_each_trials', timesteps[-1], i_eps)

    writer.close()

    end_time_train = time.time()
    if save_freq > 0 and save_dir:
        agent.save_models(os.path.join(save_dir, str(i_eps)))

    returns = env.get_episode_rewards()
    print("Length of returns:", len(returns))
    print("Length of timesteps:", len(timesteps))
    print("Length of goals:", len(goals))
    save_path = os.path.join(dir, title + "{}".format(str(seed)))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(os.path.join(dir, title + "{}".format(str(seed))), np.column_stack((returns, timesteps, goals)))

    log_dir = os.path.split(os.path.abspath(__file__))[0] + "/logs/eval/" + title + "__" + str(seed) + " " + SEQUENCE
    writer_eval = SummaryWriter(log_dir)
    if evaluation_episodes > 0:
        print("Evaluating agent over {} episodes".format(evaluation_episodes))
        agent.epsilon_final = 0.
        agent.epsilon = 0.
        agent.noise = None
        agent.actor.eval()
        agent.actor_param.eval()
        start_time_eval = time.time()
        evaluation_results = evaluate(env, agent, evaluation_episodes, writer_eval)  # returns, timesteps, goals
        end_time_eval = time.time()
        print("Ave. evaluation return =", sum(evaluation_results[:, 0]) / evaluation_results.shape[0])
        print("Ave. timesteps =", sum(evaluation_results[:, 1]) / evaluation_results.shape[0])
        goal_timesteps = evaluation_results[:, 1][evaluation_results[:, 2] == 1]
        if len(goal_timesteps) > 0:
            print("Ave. timesteps per goal =", sum(goal_timesteps) / evaluation_results.shape[0])
        else:
            print("Ave. timesteps per goal =", sum(goal_timesteps) / evaluation_results.shape[0])
        print("Ave. goal prob. =", sum(evaluation_results[:, 2]) / evaluation_results.shape[0])
        np.save(os.path.join(dir, title + "{}e".format(str(seed))), evaluation_results)
        print("Evaluation time: %.2f seconds" % (end_time_eval - start_time_eval))
    print("Training time: %.2f seconds" % (end_time_train - start_time_train))

    #print(agent)
    env.close()


def compute_n_step_returns(episode_transitions, gamma):
    n = len(episode_transitions)
    n_step_returns = np.zeros((n,))
    n_step_returns[n - 1] = episode_transitions[n - 1][2]  # Q-value is just the final reward
    for i in range(n - 2, 0, -1):
        reward = episode_transitions[i][2]
        target = n_step_returns[i + 1]
        n_step_returns[i] = reward + gamma * target
    return n_step_returns

if __name__ == '__main__':
    cfg = get_args()
    train(seed=cfg.seed, episodes=80000, batch_size=cfg.batch_size, gamma=cfg.gamma,
            inverting_gradients=False, initial_memory_threshold=0, replay_memory_size=1e5, epsilon_steps=cfg.epsilon_steps,
            learning_rate_actor=cfg.actor_lr, learning_rate_actor_param=cfg.param_net_lr, title="multilane_ramp",
            epsilon_start=cfg.epsilon_initial, epsilon_final=cfg.epsilon_final, clip_grad=0.5, beta=0.0,
            scale_actions=False, evaluation_episodes=cfg.eval_eps, update_ratio=1, save_freq=2000, save_dir=cfg.save_dir,
            layers=cfg.layers_actor)

