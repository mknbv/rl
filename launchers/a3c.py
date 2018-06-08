import argparse
from os.path import join
from os import environ
import sys
from time import sleep

import gym
import numpy as np
import rl
import tensorflow as tf

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
environ["OPENBLAS_NUM_THREADS"] = '1'
environ["MKL_NUM_THREADS"] = '1'
environ["OMP_NUM_THREADS"] = '1'
environ["NUMEXPR_NUM_THREADS"] = '1'


def get_args():
  parser = rl.launch_utils.base_parser(num_timesteps=int(250e6),
                                       batch_size=5)
  parser = rl.launch_utils.distributed_parser(parser)
  parser.add_argument("--start-learning-rate", type=float, default=None)
  parser.add_argument("--learning-rate-decay-steps",
                      type=rl.launch_utils.scint, default=int(250e6))
  parser.add_argument("--end-learning-rate", type=float, default=0)
  parser.add_argument("--perform", action="store_true")
  args = parser.parse_args()
  return args


def make_env(env_id, seed, rank=0, evaluation=False):
  env = gym.make(env_id)
  env.seed(seed + rank)
  env = rl.env_wrappers.nature_dqn_wrap(env, lazy=False,
                                        clip_reward=not evaluation)
  env = rl.env_wrappers.Logging(env)
  return env


def sample_log_uniform(low=1e-4, high=1e-2):
  return np.exp(np.random.uniform(np.log(low), np.log(high)))


def main():
  args = get_args()
  if args.start_learning_rate is None:
    sys.argv.append("--start-learning-rate {}".format(sample_log_uniform()))
  distributed_setup = rl.launch_utils.distributed_setup(args)
  if args.perform:
    perform(args.env_id, args.env_seed, args.checkpoint)
    return

  if distributed_setup.job_name is None:
    return

  if distributed_setup.job_name == "ps":
    rl.launch_utils.dump_args(args)
    distributed_setup.server.join()
    return

  with tf.device(distributed_setup.device_setter):
    global_step = tf.train.create_global_step()
    learning_rate = tf.train.polynomial_decay(
        learning_rate=args.start_learning_rate,
        global_step=global_step,
        decay_steps=args.learning_rate_decay_steps,
        end_learning_rate=args.end_learning_rate
    )
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate=learning_rate,
        decay=0.99,
        epsilon=0.1
    )
    alive_workers = tf.get_variable(
        "alive_workers", shape=[distributed_setup.num_workers],
        dtype=tf.bool, initializer=tf.constant_initializer(False), trainable=False
    )
    empty_alive_workers = alive_workers.assign(
        np.zeros(distributed_setup.num_workers, dtype=np.bool)
    )
    set_alive_worker = alive_workers[distributed_setup.worker_id].assign(True)
    # Add control dep so that running [set_alive_worker, alive_workers]
    # set that the worker is alive first.
    with tf.control_dependencies([set_alive_worker]):
      alive_workers = tf.identity(alive_workers)

  env = make_env(args.env_id, args.env_seed, rank=distributed_setup.worker_id)
  eval_env = make_env(args.env_id, args.env_seed, evaluation=True)
  global_policy, local_policy =\
      rl.policies.A3CAtariPolicy.global_and_local_instances(
          env.observation_space, env.action_space,
  )

  interactions_producer = rl.data.OnlineInteractionsProducer(
      env, local_policy, batch_size=args.batch_size)
  algorithm = rl.algorithms.A3CAlgorithm(
      interactions_producer,
      global_policy=global_policy,
      local_policy=local_policy,
  )

  algorithm.build(optimizer,
                  worker_device=distributed_setup.worker_device,
                  device_setter=distributed_setup.device_setter)

  summary_manager = rl.trainers.SummaryManager(
      logdir=join(args.logdir, "w-{}".format(distributed_setup.worker_id)),
      summary_period=args.summary_period
  )
  trainer = rl.trainers.DistributedTrainer(
      target=distributed_setup.server.target,
      is_chief=(distributed_setup.worker_id == 0),
      summary_manager=summary_manager,
      checkpoint_dir=args.logdir,
      checkpoint_period=args.checkpoint_period,
      checkpoint=args.checkpoint,
      config=tf.ConfigProto(
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1,
        allow_soft_placement=False
      )
  )

  with trainer.managed_session() as sess:
    assert all(v.device == "/job:ps/task:0" for v in  optimizer.variables()),\
        "\n".join(v.device for v in optimizer.variables())
    values = sess.run({
      "step": global_step,
      "sync_ops": algorithm.sync_ops,
      "set_alive_worker": set_alive_worker,
      "alive_workers": alive_workers
    })

    while not np.all(values["alive_workers"]):
      sleep(1)
      values["alive_workers"] = sess.run(alive_workers)

    last_eval_step = values["step"] - values["step"] % args.eval_period
    if last_eval_step == 0:
      last_eval_step = None

    if distributed_setup.worker_id != 1:
      interactions_producer.start(summary_manager.summary_writer,
                                  args.summary_period,
                                  sess)
    while not sess.should_stop() and values["step"] < args.num_timesteps:
      if args.worker_id == 1:
        values = sess.run({
          "step": global_step,
          "sync_ops": algorithm.sync_ops,
          "alive_workers": alive_workers,
          "set_alive_worker": set_alive_worker,
        })
        if last_eval_step is None\
            or values["step"] - last_eval_step >= args.eval_period:
          rl.launch_utils.evaluate(eval_env, local_policy,
                                   summary_manager.summary_writer,
                                   step=values["step"], sess=sess)
          last_eval_step = values["step"] - values["step"] % args.summary_period
          summary = tf.Summary()
          summary.value.add(tag="Diagnostics/num_alive_workers",
                            simple_value=np.sum(values["alive_workers"]))
          summary_manager.summary_writer.add_summary(summary,
                                                     global_step=values["step"])
          sess.run(empty_alive_workers)
        else:
          sleep(60)

      else:
        values["step"] = trainer.step(algorithm, fetches=[set_alive_worker])[0]


if __name__ == "__main__":
  main()
