#!/bin/bash

echo -e "Learning A*_hat using segment legnth of 2.\n"
python3 -m learn_advantage.advantage_learning_exp_handler --LR 2 --N_ITERS 1000 --preference_assum pr --num_prefs "10,100,100" --end_MDP 130 --seg_length 2 --output_dir_prefix "data/results/LOOP_EXP_"  --MDP_dir data/input/fig_5_random_MDPs/ --extra_details "two_trans"
python3 -m learn_advantage.advantage_learning_exp_handler --LR 2 --N_ITERS 1000 --preference_assum pr --num_prefs "10,100,100" --end_MDP 130 --mode sigmoid --seg_length 2 --output_dir_prefix "data/results/LOOP_EXP_"  --MDP_dir data/input/fig_5_random_MDPs/ --extra_details "two_trans"

echo -e "Learning A*_hat using segment legnth of 1.\n"
python3 -m learn_advantage.advantage_learning_exp_handler --LR 2 --N_ITERS 1000 --preference_assum pr --num_prefs "10,100,100" --end_MDP 130 --seg_length 1 --output_dir_prefix "data/results/LOOP_EXP_" --MDP_dir data/input/fig_5_random_MDPs/ --extra_details "single_trans"
python3 -m learn_advantage.advantage_learning_exp_handler --LR 2 --N_ITERS 1000 --preference_assum pr --num_prefs "10,100,100" --end_MDP 130 --mode sigmoid --seg_length 1 --output_dir_prefix "data/results/LOOP_EXP_"  --MDP_dir data/input/fig_5_random_MDPs/ --extra_details "single_trans" 

echo -e "Plotting results like in Figure 5.\n"
python3 -m learn_advantage.analysis.generate_fig_5 --num_prefs "10,100,100" --end_MDP 130 --MDP_dir data/fig_5_random_MDPs/ --output_dir_prefix "data/results/LOOP_EXP_" --all_extra_details "two_trans,single_trans"