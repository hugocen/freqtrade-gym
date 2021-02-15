# freqtrade-gym

This project is base on [freqtrade](https://github.com/freqtrade/freqtrade)  

The project is in very early stage, so there are a lot of inconvenient part that you have to set up manually. I am working on the improvements.   

## Installation 
### 1. freqtrade
Follow the [freqtrade documentation](https://www.freqtrade.io/en/latest/) to install freqtrade  
  
Initialize the user_directory  
```sh
freqtrade create-userdir --userdir user_data/
```  

### 2. Pandas
```sh
pip install pandas
```

### 3. OpenAI Gym
```sh
pip install gym
```

### 4. Copy freqtrade-gym files  
#### Baseline files  
IndicatorforRL.py -> [freqtrade home]/user_data/strategies/IndicatorforRL.py  
config_rl.json -> [freqtrade home]/config_rl.json  
freqtradegym.py -> [freqtrade home]/freqtradegym.py  
deep_rl.py -> [freqtrade home]/deep_rl.py  
#### RLib  
Copy first the baseline files.  
LoadRLModel.py -> [freqtrade home]/user_data/strategies/LoadRLModel.py  
rllib_example.py -> [freqtrade home]/rllib_example.py  

## Example Usage (baseline)  
The usage example is deep_rl.py and the config for freqtrade and freqtrade-gym is config_rl.json and uses IndicatorforRL.py as feature extraction.  
This demo is using [openai baseline library](https://github.com/hill-a/stable-baselines) to train reinforcement learning agents.  
Baseline can install by  
```sh
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
pip install stable-baselines[mpi]
```

Download historical data  
(Remember to download a little bit more data than the timerange in config file just in case.)  
```sh
freqtrade download-data -c <config file> --days <Int> -t {1m,3m,5m...}
```  
To match the example config_rl.json  
```sh
freqtrade download-data -c config_rl.json --timerange 20201119-20201201 -t 15m
```  

Move the IndicatorforRL.py into user_data/strategies (you should have user_data/strategies/IndicatorforRL.py)  

Run the demo to train an agent.
```sh
python deep_rl.py
```  

You can use tensorboard to monior the training process  
logdir is defined in deep_rl.py when initializing the rl model
```sh
tensorboard --logdir <logdir>
```  
This will look like  
![alt tensorboard](TensorBoardScreenshot.png?raw=true  "tensorboard")  

## Example Usage (RLlib)  
The usage example is rllib_example.py and the config for freqtrade and freqtrade-gym is config_rl.json and uses IndicatorforRL.py as feature extraction.  
This demo is using [RLlib](https://docs.ray.io/en/master/rllib.html) to train reinforcement learning agents.  
Baseline can install by  
```sh
pip install 'ray[rllib]'
```  

Run the demo to train an agent.
```sh
python rllib_example.py
```  


## Example of Loading model for backtesting or trading (baseline)  

Move the LoadRLModel.py into user_data/strategies (you should have user_data/strategies/LoadRLModel.py)  

Modified the class intial load model part to your model type and path.  

Modified the populate_indicators and rl_model_redict method for your gym settings.  

Run the backtesting  
```sh  
freqtrade backtesting -c config_rl.json -s LoadRLModel
```  

Dry-run trading (remove --dry-run for real deal!)  
```sh  
freqtrade trade --dry-run -c config_rl.json -s LoadRLModelgProto
```


## TODO  
- [x] Update the strategy for loadinf the trained model for backtesting and real trading. (baseline)  
- [ ] The features name and total feature number(freqtradegym.py line 89) have to manually match in the indicator strategy and in freqtradegym. I would like to come up with a way to set up features in config file.  
- [x] RLlib example.
- [ ] Update the strategy for loadinf the trained model for backtesting and real trading (RLlib).
- [ ] NEAT example.

# DISCLAIMER
This software is for educational purposes only. Do not risk money which you are afraid to lose. USE THE SOFTWARE AT YOUR OWN RISK. THE AUTHORS AND ALL AFFILIATES ASSUME NO RESPONSIBILITY FOR YOUR TRADING RESULTS.  
