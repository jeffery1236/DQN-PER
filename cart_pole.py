import gym
import matplotlib.pyplot as plt
import numpy as np
from deepQagent import Agent
from DDQAgent import DDQAgent
from DDQN_PER import DqnPer
from utils import plot_learning_curve, save_model, load_model


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    env.reset()

    test_mode = False
    num_games = 2500
    best_score = -np.inf
    scores = []
    eps_history = []

    # lr = 0.001
    # gamma = 0.96
    # state_dims = env.observation_space.shape[0]
    # num_actions = env.action_space.n
    # terminal_pos = 0.5
    # epsilon_min = 0.01
    # epsilon_dec = 1e-5

    agent = DqnPer(lr=0.001, gamma=0.99,
                  obs_dims=env.observation_space.shape[0],
                  num_actions=env.action_space.n,
                  epsilon_min=0.01, epsilon_dec=5e-5,
                  mem_size=1000, mini_batchsize=64,
                  env_name="cartpole", algo_name='DDQN-PER',checkpoint_dir="temp/")

    # agent = DDQAgent(lr=0.001, gamma=0.99,
    #             obs_dims=env.observation_space.shape[0],
    #             num_actions=env.action_space.n, epsilon=1,
    #             epsilon_min=0.01, epsilon_dec=5e-5,
    #             mem_size=1000, mini_batchsize=64,
    #             env_name="cartpole", algo_name='DoubleDQN',checkpoint_dir="temp/")
    
    if test_mode:
        agent.load_models()

    for count in range(num_games):
        state = env.reset()
        
        done = False
        score = 0
        while not done:
            # env.render()
            action = agent.get_action(state)
            new_state, reward, done, _ = env.step(action)

            if not test_mode:
                agent.store_memory(state, action, reward, new_state, done)
                agent.learn()

            score += reward
            state = new_state

        scores.append(score)
        eps_history.append(agent.epsilon)
        
        if count % 100 == 0:
            avg_score = np.mean(scores[-100:])
            if score > best_score:
                best_score = score
                if not test_mode:
                    save_model(agent)
                    agent.save_models()
            
            print(f'Current average score: {avg_score}')

        print(f"Episode: {count+1}, score: {score}, current epsilon: {agent.epsilon}")
    
    env.close()
    x = range(1, num_games+1)
    plot_learning_curve(x, scores, eps_history, 'cartpole_DoubleDQN-PER.png')