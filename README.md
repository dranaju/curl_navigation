# Depth-CUPRL Implementation

roslaunch hydrone_aerial_underwater_ddpg hydrone_aerial_deep_rl.launch world_name:=stage_2_aerial gui:=false

roslaunch curl_navigation curl.launch

tensorboard --logdir=evaluations
