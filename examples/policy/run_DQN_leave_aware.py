import argparse
import os
import pickle
import random
import sys
import traceback

import numpy as np
import torch

sys.path.extend([".", "./examples", "./src", "./src/DeepCTR-Torch", "./src/tianshou"])

from policy_utils import (  # noqa: E402
    get_args_all,
    learn_policy,
    prepare_dir_log,
    prepare_test_envs,
    prepare_user_model,
    setup_state_tracker,
)
from run_DQN import get_args_DQN, setup_policy_model  # noqa: E402
from src.core.envs.Simulated_Env.leave_aware import LeaveAwareSimulatedEnv  # noqa: E402
from src.core.util.data import get_env_args, get_true_env  # noqa: E402
from src.tianshou.tianshou.env import DummyVectorEnv  # noqa: E402

import logzero  # noqa: E402


def get_args_leave_aware():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="DQNLeaveAware")
    parser.add_argument("--message", type=str, default="DQN-leave-aware")
    parser.add_argument("--lambda_leave_penalty", type=float, default=1.0)
    parser.add_argument("--lambda_distance_bonus", type=float, default=0.0)
    return parser.parse_known_args()[0]


def prepare_train_envs_leave_aware(args, ensemble_models, env, kwargs_um):
    with open(ensemble_models.PREDICTION_MAT_PATH, "rb") as file:
        predicted_mat = pickle.load(file)

    kwargs = {
        "ensemble_models": ensemble_models,
        "env_task_class": type(env),
        "task_env_param": kwargs_um,
        "task_name": args.env,
        "predicted_mat": predicted_mat,
        "lambda_leave_penalty": args.lambda_leave_penalty,
        "lambda_distance_bonus": args.lambda_distance_bonus,
    }

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_envs = DummyVectorEnv([lambda: LeaveAwareSimulatedEnv(**kwargs) for _ in range(args.training_num)])
    train_envs.seed(args.seed)
    return train_envs


def main(args):
    model_save_path, logger_path = prepare_dir_log(args)

    ensemble_models = prepare_user_model(args)
    env, dataset, kwargs_um = get_true_env(args)
    train_envs = prepare_train_envs_leave_aware(args, ensemble_models, env, kwargs_um)
    test_envs_dict = prepare_test_envs(args, env, kwargs_um)

    state_tracker = setup_state_tracker(args, ensemble_models, env, train_envs, test_envs_dict)
    policy, train_collector, test_collector_set, optim = setup_policy_model(
        args, state_tracker, train_envs, test_envs_dict
    )

    learn_policy(
        args,
        env,
        dataset,
        policy,
        train_collector,
        test_collector_set,
        state_tracker,
        optim,
        model_save_path,
        logger_path,
        trainer="offpolicy",
    )


if __name__ == "__main__":
    trainer = "offpolicy"
    args_all = get_args_all(trainer)
    args = get_env_args(args_all)
    args_dqn = get_args_DQN()
    args_leave = get_args_leave_aware()
    args_all.__dict__.update(args.__dict__)
    args_all.__dict__.update(args_dqn.__dict__)
    args_all.__dict__.update(args_leave.__dict__)
    try:
        main(args_all)
    except Exception:
        var = traceback.format_exc()
        print(var)
        logzero.logger.error(var)
