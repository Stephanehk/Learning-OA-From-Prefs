#!/bin/bash

if [ "$1" == "TEST" ]; then
    echo -e "Note: Running experiments with limited data. To use the full dataset, do not include the TEST flag.\n"
    echo -e "Running all reward/optimal advantage function learning methedologies.\n"
    python3 -m learn_advantage.advantage_learning_exp_handler --LR 2 --N_ITERS 1000 --preference_assum pr --num_prefs "10000" --start_MDP 100 --end_MDP 110
    python3 -m learn_advantage.analysis.generate_fig_6 --num_prefs "10000" --start_MDP 100 --end_MDP 110 --preference_assum pr
else
    echo -e "Note: Running experiments with the full dataset. To use the a smaller representitive dataset, include the TEST flag.\n"
    echo -e "Running all reward/optimal advantage function learning methedologies.\n"
    python3 -m learn_advantage.advantage_learning_exp_handler --LR 2 --N_ITERS 1000 --preference_assum pr --num_prefs "100000" --start_MDP 100 --end_MDP 200
    python3 -m learn_advantage.analysis.generate_fig_6 --num_prefs "100000" --start_MDP 100 --end_MDP 200 --preference_assum pr
fi