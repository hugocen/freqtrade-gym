import random
import json
import logging
import copy
from datetime import datetime, timedelta
from typing import Any, Dict, List, NamedTuple, Optional
from pandas import DataFrame
import gym
from gym import spaces
import pandas as pd
import numpy as np
import talib.abstract as ta

from freqtrade.data import history
from freqtrade.data.converter import trim_dataframe
from freqtrade.configuration import (TimeRange, remove_credentials,
                                     validate_config_consistency)
from freqtrade.resolvers import ExchangeResolver, StrategyResolver
from freqtrade.exchange import timeframe_to_minutes, timeframe_to_seconds
from freqtrade.persistence import Trade
from freqtrade.strategy.interface import IStrategy, SellCheckTuple, SellType

from .env_render import TradingRender

logger = logging.getLogger(__name__)


class TradingEnv(gym.Env):
    """A trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human', 'system', 'none']}

    def __init__(self, config):
        super(TradingEnv, self).__init__()

        self.config = config
        self.strategy = StrategyResolver.load_strategy(config)
        self.fee = config['fee']
        self.timeframe = str(config.get('ticker_interval'))
        self.timeframe_min = timeframe_to_minutes(self.timeframe)
        self.required_startup = self.strategy.startup_candle_count

        data, timerange = self.load_bt_data()
        # need to reprocess data every time to populate signals
        preprocessed = self.strategy.ohlcvdata_to_dataframe(data)
        del data

        # Trim startup period from analyzed dataframe
        dfs = []
        for pair, df in preprocessed.items():
            dfs.append(trim_dataframe(df, timerange))
        del preprocessed
        self.rest_idx = set()
        idx = 0
        for d in dfs:
            idx += d.shape[0]
            self.rest_idx.add(idx)
        print(self.rest_idx)

        df = pd.concat(dfs, ignore_index=True)
        del dfs

        # setting
        df = df.dropna()
        self.pair = pair
        
        self.ticker = self._get_ticker(df)
        del df

        self.lookback_window_size = 40

        # start
        logger.info('Using stake_currency: %s ...', self.config['stake_currency'])
        logger.info('Data Length: %s ...', len(self.ticker))


        self.stake_amount = self.config['stake_amount']

        self.reward_decay = 0.0005
        self.not_complete_trade_decay = 0.5
        self.game_loss = -0.5
        self.game_win = 1.0
        self.simulate_length = self.config['simulate_length']
           
        # Actions 
        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Box(
            low=np.full(55, -np.inf), high=np.full(55, np.inf), dtype=np.float)

    def _next_observation(self):    
        row = self.ticker[self.index]

        trad_status = 0
        if self.trade != None:
            trad_status = self.trade.calc_profit_ratio(rate=row.open)

        obs = np.array([
            row.open,
            row.high,
            row.low,
            row.close,
            row.volume,
            row.adx,
            row.plus_dm,
            row.plus_di,
            row.minus_dm,
            row.minus_di,
            row.aroonup,
            row.aroondown,
            row.aroonosc,
            row.ao,
            row.kc_percent,
            row.kc_width,
            row.uo,
            row.cci,
            row.rsi,
            row.fisher_rsi,
            row.slowd,
            row.slowk,
            row.fastd,
            row.fastk,
            row.fastd_rsi,
            row.fastk_rsi,
            row.macd,
            row.macdsignal,
            row.macdhist,
            row.mfi,
            row.roc,
            row.bb_percent,
            row.bb_width,
            row.wbb_percent,
            row.wbb_width,
            row.htsine,
            row.htleadsine,
            row.CDLHAMMER,
            row.CDLINVERTEDHAMMER,
            row.CDLDRAGONFLYDOJI,
            row.CDLPIERCING,
            row.CDLMORNINGSTAR,
            row.CDL3WHITESOLDIERS,
            row.CDLHANGINGMAN,
            row.CDLSHOOTINGSTAR,
            row.CDLGRAVESTONEDOJI,
            row.CDLDARKCLOUDCOVER,
            row.CDLEVENINGDOJISTAR,
            row.CDLEVENINGSTAR,
            row.CDL3LINESTRIKE,
            row.CDLSPINNINGTOP,
            row.CDLENGULFING,
            row.CDLHARAMI,
            row.CDL3OUTSIDE,
            row.CDL3INSIDE,
        ], dtype=np.float)

        self.status = copy.deepcopy(row)
        
        return obs

    def _take_action(self, action):
        # Hold
        if action == 0:
            return
        # Buy
        if action == 1:
            if self.trade == None:
                self.trade = Trade(
                    pair=self.pair,
                    open_rate=self.status.open,
                    open_date=self.status.date,
                    stake_amount=self.stake_amount,
                    amount=self.stake_amount / self.status.open,
                    fee_open=self.fee,
                    fee_close=self.fee,
                    is_open=True,
                )
                self.trades.append({
                    "step": self.index,
                    "type": 'buy',
                    "total": self.status.open
                })

                logger.debug("{} - Backtesting emulates creation of new trade: {}.".format(
                    self.pair, self.trade))

        # Sell
        if action == 2:
            if self.trade != None:
                profit_percent = self.trade.calc_profit_ratio(rate=self.status.open)
                profit_abs = self.trade.calc_profit(rate=self.status.open)
                self.money += profit_abs
                self.trade = None
                self._reward = profit_percent

                self.trades.append({
                    "step": self.index,
                    "type": 'sell',
                    "total": self.status.open
                })

    def step(self, action):
        # Execute one time step within the environment
        self._reward = 0
        self._take_action(action)

        self.index += 1
        if self._reward > 1.5:
            self._reward = 0

        if self.index >= len(self.ticker):
            self.index = 0

        self.steps += 1

        self.total_reward += self._reward

        # done = (self._reward < self.game_loss) # or (self.steps > self.day_step)
        # done = (self.total_reward < self.game_loss) or (self.total_reward > self.game_win) or (self.steps > self.day_step)
        done = self.steps > self.simulate_length     

        obs = self._next_observation()

        return obs, self._reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.steps = 0
        self.index = random.randint(0, len(self.ticker)-1)

        self.trade = None
        self.trades = []

        self._reward = 0
        self.total_reward = 0
        self.money = 0

        self.visualization = None        

        return self._next_observation()

    def render(self, mode='live', close=False):
        # Render the environment to the screen
        print(f'Step: {self.index}')
        print(f'Reward: {self._reward}')
    
    def load_bt_data(self):
        timerange = TimeRange.parse_timerange(None if self.config.get(
            'timerange') is None else str(self.config.get('timerange')))

        data = history.load_data(
            datadir=self.config['datadir'],
            pairs=self.config['exchange']['pair_whitelist'],
            timeframe=self.timeframe,
            timerange=timerange,
            startup_candles=self.required_startup,
            fail_without_data=True,
            data_format=self.config.get('dataformat_ohlcv', 'json'),
        )

        min_date, max_date = history.get_timerange(data)

        logger.info(
            'Loading data from %s up to %s (%s days)..',
            min_date.isoformat(), max_date.isoformat(), (max_date - min_date).days
        )
        # Adjust startts forward if not enough data is available
        timerange.adjust_start_if_necessary(timeframe_to_seconds(self.timeframe),
                                            self.required_startup, min_date)

        return data, timerange
    
    def _get_ticker(self, processed: DataFrame) -> List:
        processed.drop(processed.head(1).index, inplace=True)

        return [x for x in processed.itertuples()]
    
