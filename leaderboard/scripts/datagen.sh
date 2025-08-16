#!/bin/bash
WORK_DIR=${1:-/data/ITS_2025/tfddcarla}
CARLA_ROOT=${2:-/data/ITS_2025/tfddcarla/carla}
TOTAL_REPS=${3:-2}  # how many times to loop through all scenarios
echo "$WORK_DIR"
CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg

SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}

# Keep leaderboard repetitions at 0
export LB_REPETITIONS=1
export CHALLENGE_TRACK_CODENAME=MAP
export TEAM_AGENT=${WORK_DIR}/team_code_autopilot/data_agent.py
export DEBUG_CHALLENGE=0
export RESUME=1
export DATAGEN=1
 
# Outer loop for repetitions
for REP in $(seq 1 $TOTAL_REPS); do
    echo "==============================="
    echo " Starting repetition $REP of $TOTAL_REPS"
    echo "==============================="

    # Loop through all ScenarioN folders in scenarios
    for SCENARIO_DIR in "${WORK_DIR}"/leaderboard/data/training/scenarios/Scenario*; do
        SCENARIO_NUM=$(basename "$SCENARIO_DIR" | sed 's/Scenario//')
        ROUTE_DIR="${WORK_DIR}/leaderboard/data/training/routes/Scenario${SCENARIO_NUM}"

        # Loop through every Town* file in this scenario folder
        for SCENARIO_FILE in "$SCENARIO_DIR"/Town*; do
            BASENAME=$(basename "$SCENARIO_FILE" .json)
            ROUTE_FILE="${ROUTE_DIR}/${BASENAME}.xml"

            # Skip if route file doesn't exist
            if [[ ! -f "$ROUTE_FILE" ]]; then
                echo "⚠️  Skipping $BASENAME (no matching route file)"
                continue
            fi

            # Append repetition number to folder & file names
            export SAVE_PATH=${WORK_DIR}/results/${BASENAME}_run${REP}
            mkdir -p "$SAVE_PATH"

            export SCENARIOS="$SCENARIO_FILE"
            export ROUTES="$ROUTE_FILE"
            export CHECKPOINT_ENDPOINT="${SAVE_PATH}/${BASENAME}_run${REP}.json"

            echo "-------------------------------"
            echo " Running ${BASENAME} (Repetition ${REP})"
            echo " SCENARIO_FILE: $SCENARIO_FILE"
            echo " ROUTE_FILE: $ROUTE_FILE"
            echo " SAVE_PATH: $SAVE_PATH"
            echo " CHECKPOINT: $CHECKPOINT_ENDPOINT"
            echo "-------------------------------"


            python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator_local.py \
              --scenarios="${SCENARIOS}" \
              --routes="${ROUTES}" \
              --repetitions=${LB_REPETITIONS} \
              --track=${CHALLENGE_TRACK_CODENAME} \
              --checkpoint="${CHECKPOINT_ENDPOINT}" \
              --agent="${TEAM_AGENT}" \
              --agent-config="${TEAM_CONFIG}" \
              --debug=${DEBUG_CHALLENGE} \
              --resume=${RESUME}
        done
    done
done
