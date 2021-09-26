from pathlib import Path
from freqtrade.configuration import Configuration

import numpy as np
import datetime
import pickle

from functools import partial

import neat
import visualize
from neat.parallel import ParallelEvaluator

# Initialize empty configuration object
fconfig = Configuration.from_files(['config_rl.json'])

# Define some constants
fconfig["ticker_interval"] = "5m"
# Name of the strategy class
fconfig["strategy"] = "IndicatorforRLFull"
# Location of the data
data_location = Path(fconfig['user_data_dir'], 'data', 'binance')
# Pair to analyze - Only use one pair here

from gym_env.trading_env import TradingEnv
# from stable_baselines.deepq.policies import MlpPolicy, LnMlpPolicy

fconfig['fee'] = 0.0015
fconfig['timerange'] = '20170101-20200401'
# fconfig['pair_whitelist'] = ["BTC/USDT"]
fconfig['simulate_length'] = 60*24*30//5

n = 3

test_n = 100
TEST_MULTIPLIER = 1
T_STEPS = 10000
TEST_REWARD_THRESHOLD = None

ENVIRONMENT_NAME = None
CONFIG_FILENAME = "./neat_config"

NUM_WORKERS = 14
CHECKPOINT_GENERATION_INTERVAL = 1
CHECKPOINT_PREFIX = "./neat_checkpoints/"
GENERATE_PLOTS = False
# CHECKPOINT_FILE = "neat_checkpoints/798"
CHECKPOINT_FILE = None


PLOT_FILENAME_PREFIX = None
MAX_GENS = 400
RENDER_TESTS = False

env = None

config = None

def _eval_genomes(eval_single_genome, genomes, neat_config):
    parallel_evaluator = ParallelEvaluator(NUM_WORKERS, eval_function=eval_single_genome)

    parallel_evaluator.evaluate(genomes, neat_config)

def _run_neat(checkpoint, eval_network, eval_single_genome):
    # Create the population, which is the top-level object for a NEAT run.

    print_config_info()

    if checkpoint is not None:
        print("Resuming from checkpoint: {}".format(checkpoint))
        p = neat.Checkpointer.restore_checkpoint(checkpoint)
    else:
        print("Starting run from scratch")
        p = neat.Population(config)

    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    p.add_reporter(neat.Checkpointer(CHECKPOINT_GENERATION_INTERVAL, filename_prefix=CHECKPOINT_PREFIX))

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(False))

    # Run until a solution is found.
    winner = p.run(partial(_eval_genomes, eval_single_genome), n=MAX_GENS)

    with open('winner-feedforward', 'wb') as f:
        pickle.dump(winner, f)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    net = neat.nn.FeedForwardNetwork.create(winner, config)

    test_genome(eval_network, net)

    generate_stat_plots(stats, winner)

    print("Finishing...")

def generate_stat_plots(stats, winner):
    if GENERATE_PLOTS:
        print("Plotting stats...")
        visualize.draw_net(config, winner, view=False, node_names=None, filename=PLOT_FILENAME_PREFIX + "net")
        visualize.plot_stats(stats, ylog=False, view=False, filename=PLOT_FILENAME_PREFIX + "fitness.svg")
        visualize.plot_species(stats, view=False, filename=PLOT_FILENAME_PREFIX + "species.svg")


def test_genome(eval_network, net):
    reward_goal = config.fitness_threshold if not TEST_REWARD_THRESHOLD else TEST_REWARD_THRESHOLD

    print("Testing genome with target average reward of: {}".format(reward_goal))

    rewards = np.zeros(test_n)

    for i in range(test_n * TEST_MULTIPLIER):

        print("--> Starting test episode trial {}".format(i + 1))
        observation = env.reset()
        action = eval_network(net, observation)

        done = False
        t = 0

        reward_episode = 0

        while not done:

            if RENDER_TESTS:
                env.render()

            observation, reward, done, info = env.step(action)

            # print("\t Observation {}: {}".format(t, observation))
            # print("\t Info {}: {}".format(t, info))

            action = eval_network(net, observation)

            reward_episode += reward

            # print("\t Reward {}: {}".format(t, reward))

            t += 1

            if done:
                print("<-- Test episode done after {} time steps with reward {}".format(t + 1, reward_episode))
                pass

        rewards[i % test_n] = reward_episode

        if i + 1 >= test_n:
            average_reward = np.mean(rewards)
            print("Average reward for episode {} is {}".format(i + 1, average_reward))
            if average_reward >= reward_goal:
                print("Hit the desired average reward in {} episodes".format(i + 1))
                break


def print_config_info():
    # print("Running environment: {}".format(env.spec.id))
    print("Running with {} workers".format(NUM_WORKERS))
    print("Running with {} episodes per genome".format(n))
    print("Running with checkpoint prefix: {}".format(CHECKPOINT_PREFIX))
    print("Running with {} max generations".format(MAX_GENS))
    print("Running with test rendering: {}".format(RENDER_TESTS))
    print("Running with config file: {}".format(CONFIG_FILENAME))
    print("Running with generate_plots: {}".format(GENERATE_PLOTS))
    print("Running with test multiplier: {}".format(TEST_MULTIPLIER))
    print("Running with test reward threshold of: {}".format(TEST_REWARD_THRESHOLD))


def run(eval_network, eval_single_genome, environment_name):
    global ENVIRONMENT_NAME
    global CONFIG_FILENAME
    global env
    global config
    global CHECKPOINT_PREFIX
    global PLOT_FILENAME_PREFIX

    ENVIRONMENT_NAME = environment_name

    checkpoint = CHECKPOINT_FILE

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         CONFIG_FILENAME)

    if CHECKPOINT_PREFIX is None:
        timestamp = datetime.datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S')
        CHECKPOINT_PREFIX = "cp_" + CONFIG_FILENAME.lower() + "_" + timestamp + "_gen_"

    if PLOT_FILENAME_PREFIX is None:
        timestamp = datetime.datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S')
        PLOT_FILENAME_PREFIX = "plot_" + CONFIG_FILENAME.lower() + "_" + timestamp + "_"

    _run_neat(checkpoint, eval_network, eval_single_genome)

def eval_network(net, net_input):
    assert (len(net_input == 55))

    result = np.argmax(net.activate(net_input))

    assert (result == 0 or result == 1 or result == 2)

    return result

def eval_single_genome(genome, genome_config):
    net = neat.nn.FeedForwardNetwork.create(genome, genome_config)
    total_reward = 0.0

    for i in range(n):
        # print("--> Starting new episode")
        observation = env.reset()

        action = eval_network(net, observation)

        done = False

        while not done:

            # env.render()

            observation, reward, done, info = env.step(action)

            # print("\t Reward {}: {}".format(t, reward))

            action = eval_network(net, observation)

            total_reward += reward
            # total_reward = reward

            if done:
                # print("<-- Episode finished after {} timesteps".format(t + 1))
                break

    return total_reward / n

if __name__ == "__main__":
    env = TradingEnv(fconfig)
    obs = env.reset()

    # config["pair_whitelist"] = ["ETH/USDT"]
    # env = TradingEnv(config)
    # obs = env.reset()
    # while True:
    #     action, _states = model.predict(obs)
    #     obs, rewards, done, info = env.step(action)
    #     # obs, rewards, done, info = env.step(0)
    #     env.render()

    run(eval_network, eval_single_genome, environment_name="CartPole-v1")

