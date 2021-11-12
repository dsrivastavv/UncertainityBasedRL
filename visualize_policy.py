import numpy as np
import time
import gym
import tensorflow as tf
from baselines.common.atari_wrappers import NoopResetEnv, FrameStack
from wrappers import MontezumaInfoWrapper, make_mario_env, make_robo_pong, make_robo_hockey, \
    make_multi_pong, AddRandomStateToInfo, MaxAndSkipEnv, ProcessFrame84, ExtraTimeLimit

# taken from make_env_all_params(run.py)
def make_env(args):
    if args["env_kind"] == 'atari':
        env = gym.make(args['env'])
        assert 'NoFrameskip' in env.spec.id
        env = NoopResetEnv(env, noop_max=args['noop_max'])
        env = MaxAndSkipEnv(env, skip=4)
        env = ProcessFrame84(env, crop=False)
        env = FrameStack(env, 4)
        env = ExtraTimeLimit(env, args['max_episode_steps'])
        if 'Montezuma' in args['env']:
            env = MontezumaInfoWrapper(env)
        env = AddRandomStateToInfo(env)
    elif args["env_kind"] == 'mario':
        env = make_mario_env()
    elif args["env_kind"] == "retro_multi":
        env = make_multi_pong()
    elif args["env_kind"] == 'robopong':
        if args["env"] == "pong":
            env = make_robo_pong()
        elif args["env"] == "hockey":
            env = make_robo_hockey()
    return env


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4',
                        type=str)
    parser.add_argument('--max_episode_steps', help='maximum number of timesteps for episode', default=4500, type=int)
    parser.add_argument('--env_kind', type=str, default="atari")
    parser.add_argument('--noop_max', type=int, default=30)
    parser.add_argument('--meta_graph', type=str, default='./logs/UBRL-20211111-004523/save_net2000.ckpt.meta/')
    parser.add_argument('--ckpt_dir', type=str, default='./logs/UBRL-20211111-004523/')
    args = parser.parse_args().__dict__

    sess = tf.Session()
    saver = tf.train.import_meta_graph(args['meta_graph'])
    saver.restore(sess,tf.train.latest_checkpoint(args['ckpt_dir']))
    graph = tf.get_default_graph()
    ph_ob = graph.get_tensor_by_name('pol/ob:0') # placeholder for observation
    a_samp = graph.get_tensor_by_name('pol/ArgMax:0') # action smapled from policy

    env = make_env(args)
    state = env.reset()
    for _ in range(10000):
        env.render()
        time.sleep(.05)
        a = sess.run([a_samp],
                            feed_dict={ph_ob: state[None, None, :]})
        state, reward, done, info = env.step(a)
        if done:
            print ("done")
            break
    env.close()


