#!/bin/bash

export RESULTS_PATH=results/
BATCH_SIZE=128
NUM_BATCHES=1
LOG_FILE="${RESULTS_PATH}/gpu_memory_matrix.csv"
TEMP_MEM_FILE=$(mktemp)

mkdir -p "$RESULTS_PATH"

# Start background GPU memory logger
(
    while true; do
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n1 >> "$TEMP_MEM_FILE"
        sleep 5
    done
) &
LOGGER_PID=$!

# Run generation with timing
start_time=$(date +%s)

mattergen-generate "$RESULTS_PATH" \
    --pretrained-name=mattergen_base \
    --batch_size="$BATCH_SIZE" \
    --num_batches="$NUM_BATCHES" \
    --record-trajectories=False

end_time=$(date +%s)
elapsed=$(( end_time - start_time ))

# Stop logger
kill $LOGGER_PID
wait $LOGGER_PID 2>/dev/null

# Read memory samples into comma-separated line
MEMORY_SAMPLES=$(paste -sd, "$TEMP_MEM_FILE")

# Write header if file does not exist
if [ ! -f "$LOG_FILE" ]; then
    HEADER="BATCH_SIZE"
    n_fields=$(echo "$MEMORY_SAMPLES" | awk -F',' '{print NF}')
    for ((i=1; i<=n_fields; i++)); do
        HEADER+=",t$((i*5))s"
    done
    echo "$HEADER" > "$LOG_FILE"
fi

# Append results
echo "$BATCH_SIZE,$MEMORY_SAMPLES" >> "$LOG_FILE"

# Clean up
rm "$TEMP_MEM_FILE"

echo "Elapsed time: ${elapsed} seconds"
echo "Logged GPU memory usage to: $LOG_FILE"
