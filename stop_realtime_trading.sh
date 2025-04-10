#!/bin/bash

# Stop real-time trading platform script
# This script stops both the real-time trading agent and the real-time model trainer

echo ""
echo "========================================================================"
echo "                  STOPPING REAL-TIME TRADING PLATFORM                  "
echo "========================================================================"
echo ""

# Check if PID file exists
if [ -f .running_pids ]; then
    # Read PIDs from file
    PIDS=$(cat .running_pids)
    
    # Stop each process
    for PID in $PIDS; do
        if ps -p $PID > /dev/null; then
            echo "Stopping process with PID: $PID"
            kill $PID
        else
            echo "Process with PID $PID is not running"
        fi
    done
    
    # Remove PID file
    rm -f .running_pids
    echo "All processes stopped."
else
    echo "No running processes found. (.running_pids file not found)"
    
    # Try to find them by name
    echo "Looking for running Python processes..."
    
    MODEL_TRAINER_PID=$(ps aux | grep "python realtime_model_trainer.py" | grep -v grep | awk '{print $2}')
    TRADING_AGENT_PID=$(ps aux | grep "python run_trading_agent_realtime.py" | grep -v grep | awk '{print $2}')
    
    if [ ! -z "$MODEL_TRAINER_PID" ]; then
        echo "Found model trainer process (PID: $MODEL_TRAINER_PID). Stopping..."
        kill $MODEL_TRAINER_PID
    fi
    
    if [ ! -z "$TRADING_AGENT_PID" ]; then
        echo "Found trading agent process (PID: $TRADING_AGENT_PID). Stopping..."
        kill $TRADING_AGENT_PID
    fi
    
    if [ -z "$MODEL_TRAINER_PID" ] && [ -z "$TRADING_AGENT_PID" ]; then
        echo "No running trading processes found."
    else
        echo "Processes stopped."
    fi
fi

echo ""
echo "If there are still python processes running that should be stopped, use:"
echo "ps aux | grep python"
echo "kill -9 <PID>"
echo "" 