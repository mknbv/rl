The aim of this project is to reproduce reinforcement learning algorithms. While there are a lot of repositories that attempt to do the same, many of them may use simpler benchmarks or not achieve results that are as good as the ones reported in the papers. An additional goal is to provide a modular code base that is quite readable and reusable.

Currently implemented algorithms:
* DQN both [NIPS](https://arxiv.org/abs/1312.5602) and [Nature](https://www.nature.com/articles/nature14236)
* [QR-DQN-0,QR-DQN-1](https://arxiv.org/abs/1710.10044)
* [A3C](https://arxiv.org/abs/1602.01783) primarily based on [universe-starter-agent](https://github.com/openai/universe-starter-agent)
* [PPO](https://arxiv.org/abs/1707.06347) for recurrent and feed-forward networks

To view some of the result head over to the [wiki](https://github.com/MichaelKonobeev/rl/wiki) page.


### Installation

* Install tensorflow or tensorflow-gpu (version >= 1.7.0). [Tensorflow installation instructions](https://www.tensorflow.org/install/).

* Clone the repository:
```
git clone https://github.com/MichaelKonobeev/rl.git
```

* `cd` into the package and install it:
```
cd rl && pip install -e .
```
