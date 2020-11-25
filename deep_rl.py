from pathlib import Path
from freqtrade.configuration import Configuration

config = Configuration.from_files(['config_rl.json'])
config["ticker_interval"] = "5m"
config["strategy"] = "IndicatorforRL"

data_location = Path(config['user_data_dir'], 'data', 'binance')

from freqtradegym import TradingEnv
from stable_baselines.common.policies import MlpPolicy

from stable_baselines import ACER

if __name__ == "__main__":
    config['fee'] = 0.0015
    config['timerange'] = '20180101-20200401'
    config['simulate_length'] = 60*24*30

    env = TradingEnv(config)
    policy_kwargs = dict(layers=[512, 512, 512, 512, 512, 512])
    model = ACER(
        MlpPolicy, env,
        learning_rate=1e-4,
        policy_kwargs=policy_kwargs,
        verbose=0,
        tensorboard_log="./tensorboard/")

    model.learn(total_timesteps=int(1e+6))
    model.save('model')
