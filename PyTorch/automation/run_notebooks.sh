#!/bin/bash

# Check if the directory path is provided as an argument
if [ -z "$1" ]; then
  echo "Usage: $0 <directory-path>"
  exit 1
fi

# Get the base directory path from the argument
BASE_DIR=$1
LOG_FILE="error_log.txt"

# Clear the log file if it exists or create a new one
> "$LOG_FILE"

# Check if the given directory exists
if [ ! -d "$BASE_DIR" ]; then
  echo "Error: Directory '$BASE_DIR' does not exist." | tee -a "$LOG_FILE"
  exit 1
fi

# Loop through all subdirectories in the given directory
for DIR in "$BASE_DIR"/*/; do
  if [ -d "$DIR" ]; then
    echo "Processing directory -->: $DIR"
    
    # Check if the directory contains any .ipynb files
    IPYNB_FILES=$(find "$DIR" -maxdepth 1 -type f -name "*.ipynb")
    
    echo "All the file in this dir : $IPYNB_FILES"

    if [ -n "$IPYNB_FILES" ]; then
      # Change into the directory
      cd "$DIR" || exit
      
      # Loop through all .ipynb files in the directory
      for NOTEBOOK in $IPYNB_FILES; do
        NOTEBOOK_NAME=$(basename "$NOTEBOOK")
        echo "Running notebook: $NOTEBOOK_NAME"
        
        # Execute the notebook and log errors if any
        if ! sudo jupyter nbconvert --to notebook --execute "$NOTEBOOK_NAME" 2>>"$LOG_FILE"; then
          echo "Error executing notebook: $NOTEBOOK" | tee -a "$LOG_FILE"
        else
          echo "Successfully executed: $NOTEBOOK_NAME"
        fi
      done
      
      # Return to the base directory
      cd "$BASE_DIR" || exit
    else
      echo "No .ipynb files found in $DIR"
    fi
  fi
done

echo "Script execution completed. Check the log file at: $LOG_FILE"
