export CARLA_ROOT=/home/fypits25/Documents/tfddcarla/carla
export WORK_DIR=/home/fypits25/Documents/tfddcarla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner_tom
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard

export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/agents
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}

scenario="FollowLeadingVehicle_storm"

gnome-terminal -- bash -c "../carla/CarlaUE4.sh -quality-level=Epic "
sleep 2
gnome-terminal -- bash -c "python3 scenario_runner.py --scenario ${scenario} --reloadWorld"
sleep 2 
gnome-terminal -- bash -c "python3 manual_control.py"
