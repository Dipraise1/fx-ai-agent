#!/bin/bash

# Start real-time trading platform script
# This script starts both the real-time trading agent and the real-time model trainer

# Set the environment
export PYTHONPATH=$(pwd)

# Load environment variables if .env exists
if [ -f .env ]; then
    echo "Loading environment variables from .env file"
    # Load each line separately to avoid issues with comments
    while IFS= read -r line || [[ -n "$line" ]]; do
        # Skip comments and empty lines
        [[ "$line" =~ ^#.*$ ]] || [[ -z "$line" ]] && continue
        export "$line"
    done < .env
fi

# Parse command line arguments
DEMO_MODE=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --demo) DEMO_MODE=true ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Set demo flag if specified
if [ "$DEMO_MODE" = true ]; then
    DEMO_FLAG="--demo"
    echo "Running in DEMO mode (no API credentials required)"
else
    DEMO_FLAG=""
    # Check for API credentials
    if [ -z "$FINNHUB_API_KEY" ] || [ -z "$FINNHUB_SECRET" ]; then
        echo "Error: Missing Finnhub API credentials. Please set FINNHUB_API_KEY and FINNHUB_SECRET in .env or use --demo flag."
        exit 1
    fi
    echo "Using API credentials from environment variables"
fi

# Create a place to store log output
mkdir -p logs

echo ""
echo "========================================================================"
echo "                   STARTING REAL-TIME TRADING PLATFORM                  "
echo "========================================================================"
echo ""

# Start the model trainer in the background
echo "Starting real-time model trainer..."
python realtime_model_trainer.py $DEMO_FLAG > logs/model_trainer.log 2>&1 &
MODEL_TRAINER_PID=$!
echo "Model trainer started (PID: $MODEL_TRAINER_PID)"

# Wait a moment to ensure the model trainer has started
sleep 2

# Start the trading agent
echo "Starting real-time trading agent..."
python run_trading_agent_realtime.py $DEMO_FLAG > logs/trading_agent.log 2>&1 &
TRADING_AGENT_PID=$!
echo "Trading agent started (PID: $TRADING_AGENT_PID)"

echo ""
echo "Both processes started successfully!"
echo "- Model trainer logs: logs/model_trainer.log"
echo "- Trading agent logs: logs/trading_agent.log"
echo ""
echo "To stop all processes, run: ./stop_realtime_trading.sh"
echo "Or press Ctrl+C to stop now"
echo ""

# Create a file with the PIDs for the stop script
echo "$MODEL_TRAINER_PID $TRADING_AGENT_PID" > .running_pids

# Wait for user interrupt
trap 'echo "Stopping all processes..."; kill $MODEL_TRAINER_PID $TRADING_AGENT_PID 2>/dev/null; rm -f .running_pids; echo "All processes stopped."; exit 0' INT TERM

# Keep the script running
while true; do
    sleep 1
    # Check if processes are still running
    if ! ps -p $MODEL_TRAINER_PID > /dev/null || ! ps -p $TRADING_AGENT_PID > /dev/null; then
        echo "One or more processes have stopped unexpectedly."
        break
    fi
done

echo "Cleaning up..."
kill $MODEL_TRAINER_PID $TRADING_AGENT_PID 2>/dev/null
rm -f .running_pids
echo "All processes stopped." 