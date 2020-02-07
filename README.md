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

3. **Download assets** 

	Download and unzip the assets, configs and models folder to the root directory:
	```bash
	wget ftp://cs.stanford.edu/cs/cvgl/robovat/data.zip
	wget ftp://cs.stanford.edu/cs/cvgl/cavin/models.zip
	unzip data.zip
	unzip models.zip
	```

## Usage

### Run with pretrained models

To execute a planar pushing tasks with a trained CAVIN model, we can run:
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
         --rb_dir episodes/task_agnostic_interactions/ \
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
