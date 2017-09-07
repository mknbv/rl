import argparse
import gym
from gym.envs.atari import AtariEnv
import logging
import os
import sys
import tensorflow as tf
import time

import rl.algorithms
import rl.policies as policies
from rl.trajectory import GAE
from rl.training_managers import (SingularTrainingManager,
                                  DistributedTrainingManager)
import rl.wrappers
from train_spec import create_optimizer

gym.undo_logger_setup()
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)


def get_args():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  parser.add_argument(
      "--env-id",
      required=True
  )
  parser.add_argument(
      "--logdir",
      required=True,
  )
  parser.add_argument(
      "--policy",
      required=True,
  )
  parser.add_argument(
      "--num-workers",
      type=int
  )
  parser.add_argument(
      "--worker-id",
      type=int
  )
  parser.add_argument(
      "--job-name",
      default=None,
      choices=["ps", "worker"]
  )
  parser.add_argument(
      "--trajectory-length",
      type=int,
      default=20
  )
  parser.add_argument(
      "--entropy-coef",
      type=float,
      default=0.01
  )
  parser.add_argument(
      "--value-loss-coef",
      type=float,
      default=0.25
  )
  parser.add_argument(
      "--gamma",
      type=float,
      default=0.99
  )
  parser.add_argument(
      "--lambda_",
      type=float,
      default=1
  )
  parser.add_argument(
      "--num-train-steps",
      type=int,
      default=int(4 * 1e6)
  )
  parser.add_argument(
      "--summary-period",
      type=int,
      default=500
  )
  parser.add_argument(
      "--checkpoint-period",
      type=int,
      default=int(1e6)
  )
  parser.add_argument(
      "--checkpoint",
      default=None
  )
  parser.add_argument(
      "--dry-run",
      action="store_true"
  )
  args = parser.parse_args()
  if args.job_name == "ps" and args.num_workers is None:
    parser.error("--job-name ps requires --num-workers to be specified")
  if args.job_name == "worker" and args.worker_id is None:
    parser.error("--job-name worker requires --worker-id and --num-workers "
                 "to be specified")
  if args.worker_id is not None and args.num_workers is None:
    parser.error(
        "specifying --worker-id requires --num-workers to be specified")
  return args


def preprocess_wrap(env):
  if isinstance(env.unwrapped, AtariEnv):
    env = rl.wrappers.wrap(env, [
        rl.wrappers.UniverseStarterImageWrapper(),
        rl.wrappers.ClipRewardWrapper()
      ])
  env = rl.wrappers.LoggingWrapper()(env)
  return env


def get_server(num_workers, worker_id=None, host="localhost", port=12222):
  spec = {
      "ps": ["{}:{}".format(host, port)],
      "worker": [
        "{}:{}".format(host, port + i + 1) for i in range(num_workers)]
    }
  cluster = tf.train.ClusterSpec(spec).as_cluster_def()
  if worker_id is None:
    return tf.train.Server(cluster, job_name="ps", task_index=0)
  else:
    return tf.train.Server(cluster, job_name="worker", task_index=worker_id)


def launch_workers(num_workers, port=12222, dry_run=False):
  pre_launch_commands = [
      "kill $(lsof -i:{}-{} -t) > /dev/null".format(port, port + num_workers),
      "tmux kill-session -t async-rl",
      "tmux new-session -d -s async-rl -n ps bash",
    ]
  post_launch_commands = [
      "tmux new-window -d -t async-rl -n htop",
      "tmux send-keys -t async-rl:htop 'htop' Enter",
  ]
  launch_commands = []
  args = " ".join(arg for arg in sys.argv[1:])
  cmd = "python train.py {} --job-name ps".format(args)
  launch_commands.append("tmux send-keys -t async-rl:ps '{}' Enter".format(cmd))
  for worker_id in range(num_workers):
    window_name = "w-{}".format(worker_id)
    launch_commands.append(
        "tmux new-window -d -t async-rl -n {} bash".format(window_name))
    cmd = (
        "python train.py {} --job-name worker --worker-id {}"
          .format(args, worker_id)
    )
    launch_commands.append(
        "tmux send-keys -t async-rl:{} '{}' Enter".format(window_name, cmd))
  commands = pre_launch_commands + launch_commands + post_launch_commands
  if dry_run:
    print("\n".join(commands))
    return
  for cmd in commands:
    print(cmd)
    if "--worker_id " in cmd:
      time.sleep(1)
    os.system(cmd)


def main():
  args = get_args()
  if args.job_name is None and args.num_workers is not None:
    launch_workers(args.num_workers, dry_run=args.dry_run)
    return

  if args.job_name == "ps":
    server = get_server(args.num_workers)
    while True:
      time.sleep(600)
    return

  env = preprocess_wrap(gym.make(args.env_id))
  policy_class = getattr(rl.policies, args.policy + "Policy")
  logging.info("Using {} policy".format(policy_class))

  if args.worker_id is None:
    worker_device = None
    device_setter = None
  else:
    worker_device = "job:worker/task:{}/cpu:0".format(args.worker_id)
    device_setter = tf.train.replica_device_setter(
        1, worker_device=worker_device)

  with tf.device(device_setter):
    global_step = tf.train.create_global_step()
    global_policy = policy_class(env.observation_space, env.action_space,
                                 name=policy_class.__name__ + "_global")
    optimizer = create_optimizer(env, global_policy, global_step)
  with tf.device(worker_device):
    if args.worker_id is None:
      local_policy = None
      training_manager = SingularTrainingManager(
          logdir=args.logdir,
          summary_period=args.summary_period,
          checkpoint_period=args.checkpoint_period,
          checkpoint=args.checkpoint)
    else:
      local_policy = policy_class(env.observation_space, env.action_space,
                                  name=policy_class.__name__ + "_local")
      server = get_server(args.num_workers, args.worker_id)
      training_manager = DistributedTrainingManager(
          target=server.target,
          is_chief=(args.worker_id == 0),
          logdir=os.path.join(args.logdir, "worker-{}".format(args.worker_id)),
          summary_period=args.summary_period,
          checkpoint_period=args.checkpoint_period,
          checkpoint=args.checkpoint)
    advantage_estimator = rl.trajectory.GAE(
        policy=local_policy or global_policy,
        gamma=args.gamma, lambda_=args.lambda_)
    algorithm = rl.algorithms.BaseA3CAlgorithm(
      env,
      trajectory_length=args.trajectory_length,
      global_policy=global_policy,
      local_policy=local_policy,
      advantage_estimator=advantage_estimator,
      entropy_coef=args.entropy_coef,
      value_loss_coef=args.value_loss_coef)
  algorithm.train(
      optimizer=optimizer,
      num_steps=args.num_train_steps,
      training_manager=training_manager)


if __name__ == "__main__":
  main()
