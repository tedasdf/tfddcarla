export CARLA_ROOT=/home/fypits25/Documents/tfddcarla/carla
export WORK_DIR=/home/fypits25/Documents/tfddcarla
export SCENARIO_RUNNER_ROOT=/home/fypits25/Documents/tfddcarla/scenario_runner
# Build PYTHONPATH once (order matters!)
# export PYTHONPATH=${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
# export PYTHONPATH=${CARLA_ROOT}/PythonAPI:${PYTHONPATH}
# export PYTHONPATH=${CARLA_ROOT}/PythonAPI/carla:${PYTHONPATH}
# export PYTHONPATH=${CARLA_ROOT}/PythonAPI/carla/agents:${PYTHONPATH}
# export PYTHONPATH=${SCENARIO_RUNNER_ROOT}:${PYTHONPATH}
# export PYTHONPATH=${LEADERBOARD_ROOT}:${PYTHONPATH}

export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/agents
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=${SCENARIO_RUNNER_ROOT}:${PYTHONPATH}
export PYTHONPATH=${LEADERBOARD_ROOT}:${PYTHONPATH}

# scenario="FollowLeadingVehicle_1" 
route="/home/fypits25/Documents/tfddcarla/scenario_runner/srunner/data/routes_debug.xml"
scenario_file="/home/fypits25/Documents/tfddcarla/scenario_runner/srunner/data/all_towns_traffic_scenarios.json"
routeid="0"

# agent="/home/fypits25/Documents/tfddcarla/scenario_runner/srunner/autoagents/human_agent.py"
agent="/home/fypits25/Documents/tfddcarla/team_code_transfuser/submission_agent.py"
agentconfig="/home/fypits25/Documents/tfddcarla/model_ckpt/diffusiondrive"

gnome-terminal -- bash -c "${CARLA_ROOT}/CarlaUE4.sh -quality-level=Epic --world-port=2000; exec bash"
echo "Opened CARLA"

sleep 2
gnome-terminal -- bash -c "python3 ${SCENARIO_RUNNER_ROOT}/scenario_runner.py --reloadWorld --agent ${agent} --agentConfig ${agentconfig} --route ${route} ${scenario_file} ${routeid} --timeout 30; exec bash" 
echo "Opened Scenario Runner"

# sleep 2
# gnome-terminal -- bash -c "python3 ${SCENARIO_RUNNER_ROOT}/manual_control.py; exec bash"
# echo "Opened Manual Control"

# python3 scenario_runner.py --scenario FollowLeadingVehicle_1 --reloadWorld 