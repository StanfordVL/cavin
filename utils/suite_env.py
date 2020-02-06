"""Suite for loading RoboVat Environments.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tf_agents.environments import gym_wrapper
from tf_agents.environments import wrappers

from robovat import envs


def load(env_name,
         simulator=None,
         config=None,
         debug=None,
         discount=1.0,
         max_episode_steps=None,
         gym_env_wrappers=(),
         env_wrappers=(),
         spec_dtype_map=None):
    """Loads the selected environment and wraps it with the specified wrappers.
    Note that by default a TimeLimit wrapper is used to limit episode lengths
    to the default benchmarks defined by the registered environments.

    Args:
        environment_name: Name for the environment to load.
        discount: Discount to use for the environment.
        max_episode_steps: If None the max_episode_steps will be set to the
            default step limit defined in the environment's spec. No limit is
            applied if set to 0 or if there is no timestep_limit set in the
            environment's spec.
        env_wrappers: Iterable with references to wrapper classes to use on the
            gym_wrapped environment.
        spec_dtype_map: A dict that maps gym specs to tf dtypes to use as the
            default dtype for the tensors. An easy way how to configure a
            custom mapping through Gin is to define a gin-configurable function
            that returns desired mapping and call it in your Gin config file,
            for example:
            `suite_gym.load.spec_dtype_map = @get_custom_mapping()`.
    Returns:
        A PyEnvironmentBase instance.
    """
    env_class = getattr(envs, env_name)
    env = env_class(simulator=simulator,
                    config=config,
                    debug=debug)
    return wrap_env(
        env,
        discount=discount,
        max_episode_steps=max_episode_steps,
        gym_env_wrappers=gym_env_wrappers,
        env_wrappers=env_wrappers,
        spec_dtype_map=spec_dtype_map)


def wrap_env(gym_env,
             discount=1.0,
             max_episode_steps=None,
             gym_env_wrappers=(),
             time_limit_wrapper=None,
             env_wrappers=(),
             spec_dtype_map=None,
             auto_reset=True):
    """Wraps given gym environment with TF Agent's GymWrapper.
    Note that by default a TimeLimit wrapper is used to limit episode lengths
    to the default benchmarks defined by the registered environments.
    Args:
        gym_env: An instance of OpenAI gym environment.
        discount: Discount to use for the environment.
        max_episode_steps: If None the max_episode_steps will be set to the
            default step limit defined in the environment's spec. No limit is
            applied if set to 0 or if there is no timestep_limit set in the
            environment's spec.
        gym_env_wrappers: Iterable with references to wrapper classes to use
            directly on the gym environment.
        time_limit_wrapper: Wrapper that accepts (env, max_episode_steps)
            params to enforce a TimeLimit. Usually this should be left as the
            default, wrappers.TimeLimit.
        env_wrappers: Iterable with references to wrapper classes to use on the
            gym_wrapped environment.
        spec_dtype_map: A dict that maps gym specs to tf dtypes to use as the
            default dtype for the tensors. An easy way how to configure a
            custom mapping through Gin is to define a gin-configurable function
            that returns desired mapping and call it in your Gin config file,
            for example:
            `suite_gym.load.spec_dtype_map = @get_custom_mapping()`.
        auto_reset: If True (default), reset the environment automatically
            after a terminal state is reached.

    Returns:
        A PyEnvironmentBase instance.
    """
    for wrapper in gym_env_wrappers:
        gym_env = wrapper(gym_env)

    env = gym_wrapper.GymWrapper(
            gym_env,
            discount=discount,
            spec_dtype_map=spec_dtype_map,
            auto_reset=auto_reset,
    )

    if time_limit_wrapper is None:
        time_limit_wrapper = wrappers.TimeLimit

    if max_episode_steps:
        if max_episode_steps > 0:
            env = time_limit_wrapper(env, max_episode_steps)

    for wrapper in env_wrappers:
        env = wrapper(env)

    return env
