#!/bin/bash

if [ "$1" == "TEST" ]; then

    echo -e "Note: Running experiments with limited data. To use the full dataset, do not include the TEST flag.\n"

    echo -e "Running all reward/optimal advantage function learning methedologies WITHOUT transitions from the absorbing state.\n"
    python3 -m learn_advantage.advantage_learning_exp_handler --LR 2 --N_ITERS 1000 --preference_assum pr --num_prefs "1000,3000" --start_MDP 100 --end_MDP 110 --dont_include_absorbing_transitions --output_dir_prefix "data/results/NO_ABSORBING_TRANSITIONS_" --extra_details "fig_3_test"
    python3 -m learn_advantage.advantage_learning_exp_handler --LR 2 --N_ITERS 1000 --preference_assum pr --mode stochastic --num_prefs "1000,3000" --start_MDP 100 --end_MDP 110 --dont_include_absorbing_transitions --output_dir_prefix "data/results/NO_ABSORBING_TRANSITIONS_" --extra_details "fig_3_test"
    python3 -m learn_advantage.analysis.get_scaled_returns --checkpoints_of_interest 1000 --preference_assum pr --num_prefs "1000,3000" --start_MDP 100 --end_MDP 110 --output_dir_prefix "data/results/NO_ABSORBING_TRANSITIONS_" --extra_details "fig_3_test"

    echo -e "Running all reward/optimal advantage function learning methedologies WITH transitions from the absorbing state.\n"
    python3 -m learn_advantage.advantage_learning_exp_handler --LR 2 --N_ITERS 1000 --preference_assum pr --num_prefs "1000,3000" --start_MDP 100 --end_MDP 110 --output_dir_prefix "data/results/WITH_ABSORBING_TRANSITIONS_" --extra_details "fig_3_test"
    python3 -m learn_advantage.advantage_learning_exp_handler --LR 2 --N_ITERS 1000 --preference_assum pr --mode stochastic --num_prefs "1000,3000" --start_MDP 100 --end_MDP 110 --output_dir_prefix "data/results/WITH_ABSORBING_TRANSITIONS_" --extra_details "fig_3_test"
    python3 -m learn_advantage.analysis.get_scaled_returns --checkpoints_of_interest 1000 --preference_assum pr --num_prefs "1000,3000" --start_MDP 100 --end_MDP 110 --output_dir_prefix "data/results/WITH_ABSORBING_TRANSITIONS_" --extra_details "fig_3_test"
    #Run Wilcoxon test on resulting data
    echo -e "Running Wilcoxon test on resulting data.\n"
    python3 -m learn_advantage.analysis.no_absorbing_transitions_stats_tests --preference_assum pr --num_prefs "1000,3000" --start_MDP 100 --end_MDP 110 --extra_details "fig_3_test"
else
    echo -e "Note: Running experiments with the full dataset. To use the a smaller representitive dataset, include the TEST flag.\n"
    echo -e "Running all reward/optimal advantage function learning methedologies WITHOUT transitions from the absorbing state.\n"
    python3 -m learn_advantage.advantage_learning_exp_handler --LR 2 --N_ITERS 1000 --preference_assum pr --num_prefs "300,1000,3000,10000,30000,100000" --start_MDP 100 --end_MDP 130 --dont_include_absorbing_transitions --output_dir_prefix "data/results/NO_ABSORBING_TRANSITIONS_" --extra_details "fig_3_test"
    python3 -m learn_advantage.advantage_learning_exp_handler --LR 2 --N_ITERS 1000 --preference_assum pr --mode stochastic --num_prefs "300,1000,3000,10000,30000,100000" --start_MDP 100 --end_MDP 130 --dont_include_absorbing_transitions --output_dir_prefix "data/results/NO_ABSORBING_TRANSITIONS_" --extra_details "fig_3_test"
    python3 -m learn_advantage.analysis.get_scaled_returns --checkpoints_of_interest 1000 --preference_assum pr --num_prefs "300,1000,3000,10000,30000,100000" --start_MDP 100 --end_MDP 130 --output_dir_prefix "data/results/NO_ABSORBING_TRANSITIONS_" --extra_details "fig_3_test"

    echo -e "Running all reward/optimal advantage function learning methedologies WITH transitions from the absorbing state.\n"
    python3 -m learn_advantage.advantage_learning_exp_handler --LR 2 --N_ITERS 1000 --preference_assum pr --num_prefs "300,1000,3000,10000,30000,100000" --start_MDP 100 --end_MDP 130 --output_dir_prefix "data/results/WITH_ABSORBING_TRANSITIONS_" --extra_details "fig_3_test"
    python3 -m learn_advantage.advantage_learning_exp_handler --LR 2 --N_ITERS 1000 --preference_assum pr --mode stochastic --num_prefs "300,1000,3000,10000,30000,100000" --start_MDP 100 --end_MDP 130 --output_dir_prefix "data/results/WITH_ABSORBING_TRANSITIONS_" --extra_details "fig_3_test"
    python3 -m learn_advantage.analysis.get_scaled_returns --checkpoints_of_interest 1000 --preference_assum pr --num_prefs "300,1000,3000,10000,30000,100000" --start_MDP 100 --end_MDP 130 --output_dir_prefix "data/results/WITH_ABSORBING_TRANSITIONS_" --extra_details "fig_3_test"
    #Run Wilcoxon test on resulting data
    python3 -m learn_advantage.analysis.no_absorbing_transitions_stats_tests --preference_assum pr --num_prefs "300,1000,3000,10000,30000,100000" --start_MDP 100 --end_MDP 130 --extra_details "fig_3_test"
fi