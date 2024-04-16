#!/usr/bin/env bash

export JULIA_NUM_THREADS=15 



# Path to your Julia project and script
PROJECT_PATH=~/.julia/dev/GrowthModels/
JULIA_SCRIPT=./src/Neural.jl

# Define output and log directory
OUTPUT_DIR=~/.julia/dev/GrowthModels/
LOG_DIR=~/.julia/dev/GrowthModels/
mkdir -p ${OUTPUT_DIR}
mkdir -p ${LOG_DIR}

# Generate a unique identifier for this run (e.g., using date and time)
RUN_ID=$(date '+%Y%m%d-%H%M%S')

# Define the base name for output and log files
BASE_NAME="local-nnfit-${RUN_ID}"

# Redirect stdout and stderr to log file
exec > >(tee "${LOG_DIR}/${BASE_NAME}.log") 2>&1

echo "Starting local NN run at $(date)"

# Start the Julia script with the specified number of workers
julia --project=${PROJECT_PATH} \
    -t${JULIA_NUM_THREADS} ${JULIA_SCRIPT} 


echo "Finished local nn run at $(date)"

