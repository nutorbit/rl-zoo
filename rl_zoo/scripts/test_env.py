import gymnasium as gym


def main():
    env = gym.make("MountainCarContinuous-v0", render_mode="human")

    obs = env.reset()
    done = False
    
    while not done:
        action = env.action_space.sample()
        
        next_obs, reward, done, _, _ = env.step(action)
        
        print(f"obs: {obs}, action: {action}, reward: {reward}, next_obs: {next_obs}, done: {done}")
        
        obs = next_obs


if __name__ == "__main__":
    main()
