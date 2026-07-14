import argparse
import random
import sys
import traceback

import logzero
import numpy as np
import torch
from gymnasium.spaces import Discrete

sys.path.extend([".", "./src", "./src/DeepCTR-Torch", "./src/tianshou"])

from policy_utils import (
    get_args_all,
    learn_policy,
    prepare_dir_log,
    prepare_test_envs,
    prepare_user_model,
    setup_state_tracker,
)
from src.core.collector.collector import Collector
from src.core.collector.collector_set import CollectorSet
from src.core.envs.Simulated_Env.survey_aware_leave import SurveyAwareLeaveSimulatedEnv
from src.core.policy.RecPolicy import RecPolicy
from src.core.util.data import get_env_args, get_true_env
from src.tianshou.tianshou.data import PrioritizedVectorReplayBuffer, VectorReplayBuffer
from src.tianshou.tianshou.env import DummyVectorEnv
from src.tianshou.tianshou.policy import DQNPolicy
from src.tianshou.tianshou.utils.net.common import Net


def get_args_DQN():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="DQNSurveyAwareLeave")
    parser.add_argument("--target-update-freq", type=int, default=320)
    parser.add_argument("--reward-normalization", action="store_true", default=False)
    parser.add_argument("--is-double", type=bool, default=True)
    parser.add_argument("--clip-loss-grad", action="store_true", default=False)
    parser.add_argument("--prioritized-replay", action="store_true", default=False)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--message", type=str, default="DQN-survey-aware-leave")
    parser.add_argument("--lambda_leave_penalty", type=float, default=1.0)
    parser.add_argument("--survey_beta", type=float, default=1.0)
    return parser.parse_known_args()[0]


def prepare_survey_aware_train_envs(args, ensemble_models, env, kwargs_um):
    import pickle

    with open(ensemble_models.PREDICTION_MAT_PATH, "rb") as file:
        predicted_mat = pickle.load(file)

    kwargs = {
        "ensemble_models": ensemble_models,
        "env_task_class": type(env),
        "task_env_param": kwargs_um,
        "task_name": args.env,
        "predicted_mat": predicted_mat,
        "lambda_leave_penalty": args.lambda_leave_penalty,
        "survey_beta": args.survey_beta,
    }

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_envs = DummyVectorEnv(
        [lambda: SurveyAwareLeaveSimulatedEnv(**kwargs) for _ in range(args.training_num)]
    )
    train_envs.seed(args.seed)
    return train_envs


def prepare_survey_aware_train_test_envs(args, ensemble_models):
    env, dataset, kwargs_um = get_true_env(args)
    train_envs = prepare_survey_aware_train_envs(args, ensemble_models, env, kwargs_um)
    test_envs_dict = prepare_test_envs(args, env, kwargs_um)
    return env, dataset, train_envs, test_envs_dict


def setup_policy_model(args, state_tracker, train_envs, test_envs_dict):
    if args.cpu:
        args.device = "cpu"
    else:
        args.device = torch.device(
            "cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu"
        )

    net = Net(
        args.state_dim,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        device=args.device,
    ).to(args.device)
    optim_RL = torch.optim.Adam(net.parameters(), lr=args.lr)
    optim_state = torch.optim.Adam(state_tracker.parameters(), lr=args.lr)
    optim = [optim_RL, optim_state]
    policy = DQNPolicy(
        net,
        optim,
        state_tracker=state_tracker,
        discount_factor=args.gamma,
        estimation_step=args.n_step,
        target_update_freq=args.target_update_freq,
        reward_normalization=args.reward_normalization,
        is_double=args.is_double,
        clip_loss_grad=args.clip_loss_grad,
        action_space=Discrete(args.action_shape),
    )
    policy.set_eps(args.explore_eps)

    rec_policy = RecPolicy(args, policy, state_tracker)

    if args.prioritized_replay:
        buf = PrioritizedVectorReplayBuffer(
            args.buffer_size,
            buffer_num=len(train_envs),
            alpha=args.alpha,
            beta=args.beta,
        )
    else:
        buf = VectorReplayBuffer(args.buffer_size, buffer_num=len(train_envs))

    train_collector = Collector(
        rec_policy,
        train_envs,
        buffer=buf,
        exploration_noise=args.exploration_noise,
        remove_recommended_ids=args.remove_recommended_ids,
    )
    test_collector_set = CollectorSet(
        rec_policy,
        test_envs_dict,
        args.buffer_size,
        args.test_num,
        exploration_noise=args.exploration_noise,
        force_length=args.force_length,
    )
    return rec_policy, train_collector, test_collector_set, optim


def main(args):
    model_save_path, logger_path = prepare_dir_log(args)
    ensemble_models = prepare_user_model(args)
    env, dataset, train_envs, test_envs_dict = prepare_survey_aware_train_test_envs(
        args, ensemble_models
    )
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
    args_all.__dict__.update(args.__dict__)
    args_all.__dict__.update(args_dqn.__dict__)
    try:
        main(args_all)
    except Exception:
        var = traceback.format_exc()
        print(var)
        logzero.logger.error(var)
