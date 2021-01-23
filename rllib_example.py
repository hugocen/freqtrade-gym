from pathlib import Path
from freqtrade.configuration import Configuration

config = Configuration.from_files(['config_rl.json'])

from freqtradegym import TradingEnv

import ray
from ray.tune.registry import register_env
from ray.rllib.agents import ppo

def env_creator(env_config):
    return TradingEnv(config)  # return an env instance



if __name__ == "__main__":
    # env = TradingEnv(config)
    ray.init()
    register_env("my_env", env_creator)
    trainer = ppo.PPOTrainer(env="my_env")

    while True:
        print(trainer.train())