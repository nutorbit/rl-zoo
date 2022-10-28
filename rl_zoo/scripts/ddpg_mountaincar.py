import wandb
import gymnasium as gym

from rl_zoo.off_policy.ddpg import DDPG
from rl_zoo.utils.roller import OffPolicyRoller


def main():
    env = gym.make("MountainCarContinuous-v0")

    policy = DDPG(
        env.observation_space.shape[0],
        env.action_space.shape[0],
        hiddens=[24, 48],
        gamma=0.99,
        learning_rate=1e-3
    )

    wandb.init(project="rl-zoo", entity="nutorbit")

    roller = OffPolicyRoller(
        policy,
        env,
        batch_size=32,
        update_every=1,
        warm_start=5_000,
        max_timesteps_per_episode=500,
        eval_every=2_000,
        logger=wandb
    )

    # TODO: test this
    roller.run()


if __name__ == "__main__":
    main()
