import argparse
from collections import namedtuple
from logging import getLogger
import os
import sys
import time

import numpy as np
import tensorflow as tf
from tqdm import trange


ARG_DISABLE = object()
logger = getLogger("rl")


def scint(s):
  """ Integer type that allows conversion from scientific floats. """
  float_val = float(s)
  int_val = int(float_val)
  if int_val - float_val != 0:
    raise argparse.ArgumentError("invalid scint value: %s" % s)
  else:
    return int_val


def base_parser(parser=None, *,
                env_seed=0,
                num_timesteps=int(200e6),
                batch_size=int(32),
                summary_period=int(5e3),
                checkpoint_period=int(1e6),
                eval_period=int(1e6)):
  if parser is None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--env-id", required=True)
  parser.add_argument("--logdir", required=True)

  def add_if_not_disabled(*args, default=ARG_DISABLE, **kwargs):
    if default != ARG_DISABLE:
      parser.add_argument(*args, default=default, **kwargs)

  add_if_not_disabled("--env-seed", type=int, default=env_seed)
  add_if_not_disabled("--num-timesteps", type=scint, default=num_timesteps)
  add_if_not_disabled("--batch-size", type=scint, default=batch_size)
  add_if_not_disabled("--summary-period", type=scint, default=summary_period)
  add_if_not_disabled("--checkpoint-period",
                      type=scint, default=checkpoint_period)
  add_if_not_disabled("--eval-period", type=scint, default=eval_period)
  parser.add_argument("--checkpoint")
  return parser


def distributed_parser(parser=None):
  if parser is None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--num-workers", type=int)
  parser.add_argument("--job-name", choices=["ps", "worker"])
  parser.add_argument("--session-name")
  parser.add_argument("--worker-id", type=int)
  parser.add_argument("--start-port", type=int, default=12222)
  return parser


def dump_args(args):
  if not os.path.isdir(args.logdir):
    os.makedirs(args.logdir)
  with open(os.path.join(args.logdir, "args.txt"), "w") as f:
    for key, val in vars(args).items():
      f.write("{} {}\n".format(key, val))


def get_server(num_workers, worker_id=None, host="localhost",
               start_port=12222):
  spec = {
      "ps": ["{}:{}".format(host, start_port)],
      "worker": [
        "{}:{}".format(host, start_port + i + 1) for i in range(num_workers)
      ]
  }
  cluster = tf.train.ClusterSpec(spec).as_cluster_def()
  if worker_id is None:
    return tf.train.Server(cluster, job_name="ps", task_index=0)
  else:
    return tf.train.Server(cluster, job_name="worker", task_index=worker_id)


def launch_workers(session_name, launch_command,
                   num_workers, start_port=12222, dry_run=False):
  pre_launch_commands = [
      "kill $(lsof -i:{}-{} -t) > /dev/null".format(start_port,
                                                    start_port + num_workers),
      "tmux kill-session -t {}".format(session_name),
      "tmux new-session -d -s {} -n ps bash".format(session_name),
    ]
  post_launch_commands = []
  launch_commands = []
  cmd = "python {} --job-name ps".format(launch_command, session_name)
  launch_commands.append("tmux send-keys -t {}:ps '{}' Enter"
                         .format(session_name, cmd))
  for worker_id in range(num_workers):
    window_name = "w-{}".format(worker_id)
    launch_commands.append(
        "tmux new-window -d -t {} -n {} bash"
        .format(session_name, window_name))
    cmd = (
        "python {} --job-name worker --worker-id {}"
        .format(launch_command, worker_id)
    )
    launch_commands.append(
        "tmux send-keys -t {}:{} '{}' Enter"
        .format(session_name, window_name, cmd)
    )
  commands = pre_launch_commands + launch_commands + post_launch_commands
  if dry_run:
    print("\n".join(commands))
    return
  for cmd in commands:
    print(cmd)
    if "--worker_id " in cmd:
      time.sleep(1)
    os.system(cmd)


_DistributedSetup = namedtuple(
    "DistributedSetup",
    ["job_name", "server", "num_workers", "worker_id",
     "worker_device", "device_setter"])


class DistributedSetup(_DistributedSetup):
  def __new__(cls, job_name=None, server=None, num_workers=None,
              worker_id=None, worker_device=None, device_setter=None):
    return super(DistributedSetup, cls).__new__(
        cls, job_name, server, num_workers,
        worker_id, worker_device, device_setter
    )


def distributed_setup(args):
  if args.job_name not in [None, "ps", "worker"]:
    raise ValueError("Unsupported job name: {}".format(args.job_name))
  if args.job_name == "worker" and args.worker_id is None:
    raise ValueError("job_name is worker, but worker id is not specified")

  if args.num_workers is None:
    return DistributedSetup(job_name="worker")
  if args.job_name is None and args.num_workers is not None:
    if args.session_name is not None:
      session_name = args.session_name
    else:
      session_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    launch_command = " ".join(sys.argv)
    launch_workers(session_name,
                   launch_command=launch_command,
                   num_workers=args.num_workers,
                   start_port=args.start_port)
    return DistributedSetup()

  if args.job_name == "ps":
    server = get_server(args.num_workers, start_port=args.start_port)
    return DistributedSetup(job_name="ps", server=server,
                            num_workers=args.num_workers)

  assert args.job_name == "worker" and args.worker_id is not None
  server = get_server(args.num_workers, args.worker_id,
                      start_port=args.start_port)
  worker_device = "job:worker/task:{}/cpu:0".format(args.worker_id)
  device_setter = tf.train.replica_device_setter(1,
                                                 worker_device=worker_device)

  return DistributedSetup(job_name="worker", server=server,
                          num_workers=args.num_workers,
                          worker_id=args.worker_id,
                          worker_device=worker_device,
                          device_setter=device_setter)


def evaluate(env, policy, summary_writer, seeds=[0],
             nframes=int(5e5), step=None, sess=None):
  if sess is None:
    sess = tf.get_default_session()
  if step is None:
    step = sess.run(tf.train.get_global_step())
  summary = tf.Summary()

  logger.info("Evaluation starts, step #{}".format(step))
  for seed in seeds:
    env.seed(seed)
    obs = env.reset()
    rewards = [0]
    episode_lengths = [0]
    for i in trange(0, nframes, 4):
      action = policy.act(obs[None], sess=sess)["actions"][0]
      obs, rew, done, info = env.step(action)
      if "raw_reward" in info:
        rew = info["raw_reward"]
      rewards[-1] += rew
      episode_lengths[-1] += 1
      if done:
        env.reset()
      if ("real_done" in info and info["real_done"])\
          or ("real_done" not in info and done):
        rewards.append(0)
        episode_lengths.append(0)

  num_episodes = len(rewards) - 1
  if not done and len(rewards) > 1:
    rewards = rewards[:-1]
    episode_lengths = episode_lengths[:-1]

  logger.info("Evaluation finished, step #{}, mean reward: {}"
              .format(step, np.mean(rewards)))
  summary.value.add(tag="Evaluation/reward_mean",
                    simple_value=np.mean(rewards))
  summary.value.add(tag="Evaluation/reward_median",
                    simple_value=np.median(rewards))
  summary.value.add(tag="Evaluation/min_reward", simple_value=np.min(rewards))
  summary.value.add(tag="Evaluation/max_reward", simple_value=np.max(rewards))
  summary.value.add(tag="Evaluation/num_episodes", simple_value=num_episodes)
  summary.value.add(tag="Evaluation/reward_std", simple_value=np.std(rewards))
  summary.value.add(tag="Evaluation/mean_episode_length",
                    simple_value=np.mean(episode_lengths))
  summary_writer.add_summary(summary, step)

  return rewards
