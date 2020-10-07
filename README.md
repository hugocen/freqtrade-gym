# freqtrade-gym

This project is base on [freqtrade](https://github.com/freqtrade/freqtrade)

## Installation 
### 1. freqtrade
Follow the [freqtrade documentation](https://www.freqtrade.io/en/latest/) to install freqtrade

### 2. Pandas
```sh
pip install pandas
```

### 3. OpenAI Gym
```sh
pip install gym
```

### 4. (Optional) Set Up Indicators
Move the IndicatorforRL.py into user_data/strategies (you should have user_data/strategies/IndicatorforRL.py  
This is for feature extraction. You can change this to your customized indicator strategy.

## Usage
The usage example is deep_rl.py and the config for freqtrade and freqtrade-gym is config_rl.json.  
This demo is using [openai baseline library](https://github.com/hill-a/stable-baselines) to train reinforcement learning agents.  
Baseline can install by  
```sh
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
pip install stable-baselines[mpi]
```

Run the demo to train an agent.
```sh
python deep_rl.py
```