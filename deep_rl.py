from pathlib import Path
from freqtrade.configuration import Configuration

config = Configuration.from_files(['config_rl.json'])


from freqtradegym import TradingEnv
from stable_baselines.common.policies import MlpPolicy

from stable_baselines import ACER

if __name__ == "__main__":

    env = TradingEnv(config)
    policy_kwargs = dict(layers=[32, 32])
    model = ACER(
        MlpPolicy, env,
        learning_rate=1e-4,
        policy_kwargs=policy_kwargs,
        verbose=0,
        tensorboard_log="./tensorboard/")

    model.learn(total_timesteps=int(1e+6))
    model.save('model')
