# Evaluation scripts

This directory contains scripts for evaluating the performance of the models.

The relevant configs are in `conf/eval/` .

## General files

 `asp_inference_common.py`

 `common.py`

 `ndnf_mt_problog_interpretation.py`

 `problog_inference_common.py`

## Switcheroo Environment Set

Any python script with `ss_corridor` prefix is used to evaluate models on the
Switcheroo environment set.

* Tabular
    - Evaluate RL performance: `ss_corridor_tabular_multirun_eval.py`

* MLP
    - Evaluate RL performance (MDP + POMDP):
 `ss_corridor_ppo_multirun_rl_performance_eval.py`

* NDNF-MT
    - Evaluate RL performance (MDP + POMDP):
 `ss_corridor_ppo_multirun_rl_performance_eval.py`

    - MDP - ASP rules
        * Extraction: `ss_corridor_ppo_ndnf_multirun_eval.py`
        * Inference: `ss_corridor_asp_rules_inference.py`
    - POMDP - ProbLog rules
        * Extraction: `ss_corridor_ppo_pomdp_ndnf_multirun_eval.py` and
 `ss_corridor_ppo_pomdp_ndnf_mt_post_train_interpretation.py`

        * Inference: `ss_corridor_pomdp_problog_rules_inference.py`

## Blackjack

Any python script with `blackjack` prefix is used to evaluate models on the
Blackjack environment.

* Common files
    - `blackjack_ppo_rl_eval_common.py`

* Tabular
    - Evaluate RL performance
        * Trained Q-tables: `blackjack_tabular_multirun_eval.py`
        * Table from Sutton & Barto: `blackjack_tabular_snb_eval.py`

* MLP
    - Evaluate RL performance: `blackjack_ppo_multirun_rl_performance_eval.py`
    

* NDNF-MT
    - Evaluate RL performance: `blackjack_ppo_multirun_rl_performance_eval.py`
    - ProbLog rules extraction:
`blackjack_ppo_ndnf_mt_post_train_soft_extraction.py` and
 `blackjack_ppo_ndnf_mt_post_train_interpretation.py`

    - ProbLog rules inference: `blackjack_problog_rules_inference.py`

## Taxi

Any python script with `taxi` prefix is used to evaluate models on the Taxi
environment.

* Common files
    - `taxi_ppo_rl_eval_common.py`
    - `taxi_distillation_rl_eval_common.py`
* Tabular
    - Evaluate RL performance: `taxi_tabular_multirun_eval.py`
* MLP
    - Evaluate RL performance: `taxi_ppo_multirun_rl_performance_eval.py`
* NDNF-MT (distilled)
    - Evaluate RL performance:  `taxi_distillation_multirun_rl_performance_eval.py`
    - ProbLog rules extraction:
`taxi_distillation_ndnf_mt_post_train_soft_extraction.py` and
 `taxi_distillation_ndnf_mt_post_train_interpretation.py`

    - ProbLog rules inference:  TODO

## Door Corridor

Any python script with `door_corridor` prefix is used to evaluate models on the
Door Corridor environment.

* MLP
    - Evaluate RL performance: `door_corridor_ppo_multirun_rl_performance_eval.py`
* NDNF-MT
    - Evaluate RL performance: `door_corridor_ppo_multirun_rl_performance_eval.py`
    - ASP rules extraction: `door_corridor_ppo_ndnf_mt_multirun_eval.py`
    - ASP rules inference: `door_corridor_asp_rules_inference.py`
