export CARLA_ROOT=${1:-/home/fypits25/Documents/tfddcarla/carla}
export WORK_DIR=${2:-/home/fypits25/Documents/tfddcarla}

export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}

export SCENARIOS=${WORK_DIR}/leaderboard/data/training/scenarios/ # CHANG
export ROUTES=${WORK_DIR}/leaderboard/data/training/routes/ # CHANG
export REPETITIONS=1
export CHALLENGE_TRACK_CODENAME=MAP
export CHECKPOINT_ENDPOINT=${WORK_DIR}/results/  # CHANGE
export SAVE_PATH=${WORK_DIR}/results/save/ # CHANG
export TEAM_AGENT=${WORK_DIR}/team_code_autopilot/data_agent.py
export DEBUG_CHALLENGE=0
export RESUME=0
export DATAGEN=1

python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator_local.py \
--scenarios=${SCENARIOS}  \
--routes=${ROUTES} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--resume=${RESUME}