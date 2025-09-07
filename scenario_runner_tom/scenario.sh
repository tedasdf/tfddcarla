export CARLA_ROOT=/home/fypits25/Documents/tfddcarla/carla
export WORK_DIR=/home/fypits25/Documents/tfddcarla
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner_tom

export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/agents
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg

export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}

scenario="FollowLeadingVehicle_1"

gnome-terminal -- bash -c "${CARLA_ROOT}/CarlaUE4.sh -quality-level=Epic"
echo "Opened CARLA"
sleep 2
gnome-terminal -- bash -c "python3 ${SCENARIO_RUNNER_ROOT}/scenario_runner.py --scenario ${scenario} --reloadWorld"
echo "Opened Scenario Runner"
sleep 2
gnome-terminal -- bash -c "python3 ${SCENARIO_RUNNER_ROOT}/manual_control.py"
echo "Opened Manual Control"

# python3 scenario_runner.py --scenario FollowLeadingVehicle_1 --reloadWorld