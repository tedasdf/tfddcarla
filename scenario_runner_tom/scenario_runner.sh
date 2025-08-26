export CARLA_ROOT=${1:-/home/fypits25/Documents/tfddcarla/carla}
export SCENARIO_RUNNER_ROOT=${2:-/home/fypits25/Documents/tfddcarla/scenario_runner_tom}

export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/agents
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export SCENARIO="FollowLeadingVehicle_storm"



python3 ${WORK_DIR}/scenario_runner.py \
--scenario=${SCENARIO} \
