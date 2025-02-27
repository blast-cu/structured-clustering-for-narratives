#!/bin/bash

# Define the parameter arrays
num_clusters_values=(500 600 700 800 900 1000)
w_cl_values=(0.0001 0.001 0.01 0.1 0.5 0.0 1.0 2.0)

# Count total combinations for progress tracking
total_combinations=$((${#num_clusters_values[@]} * ${#w_cl_values[@]}))
current=0

echo "Starting to run cluster.sh with all parameter combinations..."
echo "Total combinations to run: $total_combinations"
echo "----------------------------------------"

# Loop through all combinations
for num_clusters in "${num_clusters_values[@]}"; do
    for w_cl in "${w_cl_values[@]}"; do
        current=$((current + 1))
        
        # Display progress
        echo "[$current/$total_combinations] Running with num_clusters=$num_clusters, w_cl=$w_cl"
        
        # Call the cluster.sh script with the current parameters
        sbatch ./scripts/cluster.sh "$num_clusters" "$w_cl"
        
        # Check if the script executed successfully
        if [ $? -eq 0 ]; then
            echo "✓ Job scheduled successfully"
        else
            echo "✗ Failed with exit code $?"
        fi
        
        echo "----------------------------------------"
    done
done

echo "All combinations completed."