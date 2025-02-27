#!/bin/bash

# Default values
BASE_DIR="."
TOKEN_KEY="default_token"
LOG_FILE="${BASE_DIR}/error_log.txt"

# Function to display usage information
usage() {
  echo "Usage: $0 [-d base_directory] [-t token_key] [-l log_file]"
  exit 1
}

# Parse command-line options
while getopts ":d:t:l:" opt; do
  case ${opt} in
    d )
      BASE_DIR="$OPTARG"
      ;;
    t )
      TOKEN_KEY="$OPTARG"
      ;;
    l )
      LOG_FILE="$OPTARG"
      ;;
    \? )
      echo "Invalid option: -$OPTARG" >&2
      usage
      ;;
    : )
      echo "Option -$OPTARG requires an argument." >&2
      usage
      ;;
  esac
done

# Shift off the options and optional -- to get to the positional parameters
shift $((OPTIND -1))

# Display the values
echo "Base Directory: $BASE_DIR"
echo "Token Key: $TOKEN_KEY"
echo "Log File: $LOG_FILE"

# Check if the given directory exists
if [ ! -d "$BASE_DIR" ]; then
  echo "Error: Directory '$BASE_DIR' does not exist." | tee -a "$LOG_FILE"
  exit 1
fi

# Clear the log file if it exists or create a new one
: > "$LOG_FILE"

# Loop through all subdirectories in the given directory
find "$BASE_DIR" -maxdepth 1 -type d | while read -r DIR; do
  DIR_NAME="${DIR##*/}"
  echo "Processing directory: $DIR"

  # Check if the directory contains any .ipynb files
  IPYNB_FILES=$(find "$DIR" -maxdepth 1 -type f -name "*.ipynb")

  if [ -n "$IPYNB_FILES" ]; then
    echo "Found .ipynb files: $IPYNB_FILES"

    # Change into the directory
    cd "$DIR" || exit

    # Loop through all .ipynb files in the directory
    for NOTEBOOK in $IPYNB_FILES; do
      NOTEBOOK_NAME="${NOTEBOOK##*/}"
      NOTEBOOK_BASE="${NOTEBOOK_NAME%%.*}"

      echo "Running $NOTEBOOK_NAME"

      # Create a script to process the notebook
      cat <<EOF > "${DIR}/${NOTEBOOK_BASE}.sh"
#!/bin/bash

jupyter nbconvert --to python /root/${NOTEBOOK_NAME}

sed -i "s/<YOUR HUGGINGFACE TOKEN HERE>g" /root/${NOTEBOOK_BASE}.py

ipython3 /root/${NOTEBOOK_BASE}.py 2>&1 | tee /root/${NOTEBOOK_BASE}.out

if [ \$? -ne 0 ]; then
    echo "TEST FAILED" > /root/${NOTEBOOK_BASE}.out
else
    echo "TEST PASSED" > /root/${NOTEBOOK_BASE}.out
fi
EOF

      chmod +x "${DIR}/${NOTEBOOK_BASE}.sh"

      # Run the script in a Docker container
      echo "docker run --name "${NOTEBOOK_BASE}" --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none -v "${DIR}:/root" --cap-add=sys_nice --net=host --ipc=host --user root:root sapdai/gaudi-tutorial-env:latest /root/${NOTEBOOK_BASE}.sh"

      docker run --name "${NOTEBOOK_BASE}" --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none -v "${DIR}:/root" --cap-add=sys_nice --net=host --ipc=host --user root:root sapdai/gaudi-tutorial-env:latest /root/${NOTEBOOK_BASE}.sh

      if [ $? -eq 0 ] ; then
        echo "Error executing ${NOTEBOOK_BASE}" | tee -a "$LOG_FILE"
      else
        echo "Successfully executed ${NOTEBOOK_BASE}" | tee -a "$LOG_FILE"
      fi

      # Perform clean up
      docker container rm "${NOTEBOOK_BASE}"

      rm "${DIR}/${NOTEBOOK_BASE}.sh"

      rm "${DIR}/${NOTEBOOK_BASE}.py"

      echo "Finished $NOTEBOOK_NAME"
    done

    # Return to the base directory
    cd "$BASE_DIR" || exit
  else
    echo "No .ipynb files found in $DIR"
  fi
done

echo "Script execution completed. Check the log file at: $LOG_FILE"
