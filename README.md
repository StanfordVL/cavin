# Dynamics Learning with Cascaded Variational Inference for Multi-Step Manipulation
Created by [Kuan Fang](http://ai.stanford.edu/~kuanfang/), [Yuke Zhu](http://ai.stanford.edu/~yukez/), [Animesh Garg](https://www.cs.toronto.edu/~garg/), [Silvio Savarese](https://profiles.stanford.edu/intranet/silvio-savarese) and [Li Fei-Fei](https://profiles.stanford.edu/fei-fei-li)

## Citation

If you find this code useful for your research, please cite:
```
@article{fang2019cavin, 
    title={Dynamics Learning with Cascaded Variational Inference for Multi-Step Manipulation},
    author={Kuan Fang and Yuke Zhu and Animesh Garg and Silvio Savarese and Li Fei-Fei}, 
    journal={Conference on Robot Learning (CoRL)}, 
    year={2019} 
}
```

## About

This repo is an implementation of the CAVIN planner from our CoRL 2020 [paper](https://arxiv.org/abs/1910.13395). You can checkout the [project website](http://pair.stanford.edu/cavin/) for more information.

The code is based on [TF-Agent](https://github.com/tensorflow/agents). The core algorithm can be applied to any tasks designed with the [OpenAI Gym](https://github.com/tensorflow/agents) interface given the reward functions. We demonstrate CAVIN with three planar pushing tasks with three goals and constraints in simulation and the real world. The task environments are implemented in [RoboVat](https://github.com/StanfordVL/robovat/tree/master/robovat). 

## Getting Started

1. **Create a virtual environment (recommended)** 

	Create a new virtual environment in the root directory or anywhere else:
	```bash
	virtualenv --system-site-packages -p python3 .venv
	```

	Activate the virtual environment every time before you use the package:
	```bash
	source .venv/bin/activate
	```

	And exit the virtual environment when you are done:
	```bash
	deactivate
	```

2. **Install the package** 

	The package can be installed with GPU support by running:
	```bash
	pip install -r requirements_gpu.txt
	```

	Or for CPU-only:
	```bash
	pip install -r requirements_cpu.txt
	```

  Install [robovat](https://github.com/StanfordVL/robovat).

  <em>Note:</em> The code was developed with PyBullet 1.8.0. Newer versions of PyBullet might lead to different simulation results.

3. **Download assets** 

	Download and unzip the assets, configs and models folder to the root directory:
	```bash
	wget ftp://cs.stanford.edu/cs/cvgl/robovat/assets.zip
	wget ftp://cs.stanford.edu/cs/cvgl/robovat/configs.zip
	wget ftp://cs.stanford.edu/cs/cvgl/cavin/models.zip
	unzip data.zip
	unzip models.zip
	```

## Usage

### Run with pretrained models

To execute an planar pushing task (e.g. crossing) with a trained CAVIN model, we can run:
```bash
python run_env.py \
         --env PushEnv \
         --env_config configs/envs/push_env.yaml \
         --policy_config configs/policies/push_policy.yaml \
         --config_bindings "{'MAX_MOVABLE_BODIES':3,'NUM_GOAL_STEPS':3,'TASK_NAME':'crossing','LAYOUT_ID':0}" \
         --policy CavinPolicy --checkpoint models/baseline_20191001_cavin/ \
         --debug 1
```

### Data collection and training

We suggest running the data collection script in parallel on multiple CPU clusters, since the data collection may take around 10k-20k CPU hours. To collect task agnostic interactions using the heuristic pushing policy:
```bash
python tfrecord_collect.py \
         --env PushEnv \
         --env_config configs/envs/push_env.yaml \
         --policy HeuristicPushPolicy \
         --policy_config configs/policies/push_policy.yaml \
         --rb_dir episodes/task_agnostic_interactions/
```

Some of the collected files might be corrupted due to unexpected termination of the running script. To filter the corrupted files:
```bash
python filter_corrupted_tfrecords.py --data episodes/task_agnostic_interactions/
```

Before training, split the data into `train` and `eval` folders.

### Training

To train the CAVIN model on the collected data:
```bash
python tfrecord_train_eval.py \
         --env PushEnv \
         --env_config configs/envs/push_env.yaml \
         --policy_config configs/policies/push_policy.yaml \
         --rb_dir episodes/task_agnostic_interactions/ \
         --agent cavin \
         --working_dir models/YOUR_MODEL_NAME
```

To run in different tasks, we can set different values in `--config_bindings`. Specifically, `'TASK_NAME'` can be set to `'clearing'`, `'insertion'` or `'crossing'` and `'LAYOUT_ID'` can be set to `0`, `1` or `2`.
