#!/bin/bash

# Default values
BASE_DIR="."
TOKEN_KEY="default_token"
LOG_FILE="${BASE_DIR}/error_log.txt"
TIMESTAMP=$(date +"%d-%m_%H-%M")
AUTOMATION_DIR="${BASE_DIR}/${TIMESTAMP}"

# Function to display usage information
usage() {
  echo "Usage: $0 [-d base_directory] [-t token_key] [-l log_file]"
  exit 1
}

# Parse command-line options
while getopts ":d:t:l:" opt; do
  case ${opt} in
    d ) BASE_DIR="$OPTARG" ;;
    t ) TOKEN_KEY="$OPTARG" ;;
    l ) LOG_FILE="$OPTARG" ;;
    \? ) echo "Invalid option: -$OPTARG" >&2; usage ;;
    : ) echo "Option -$OPTARG requires an argument." >&2; usage ;;
  esac
done

shift $((OPTIND -1))

# Create timestamped automation directory
mkdir -p "$AUTOMATION_DIR"

# Clear or create log file
: > "$LOG_FILE"

# Find and copy runnable .ipynb files to automation directory (exclude RAG, TGI, vllms)
find "$BASE_DIR" -type d \( -name "RAG_Application" -o -name "vLLM_Tutorials" -o -name "TGI_Gaudi_tutorial" \) -prune -o -type f -name "*.ipynb" -exec cp {} "$AUTOMATION_DIR" \;


# Process notebooks in automation directory
for NOTEBOOK in "$AUTOMATION_DIR"/*.ipynb; do
  [ -e "$NOTEBOOK" ] || continue  # Skip if no notebooks found

  NOTEBOOK_NAME="$(basename "$NOTEBOOK")"
  NOTEBOOK_BASE="${NOTEBOOK_NAME%%.*}"

  echo "Running notebook: $NOTEBOOK_NAME"

  # Create script to run in Docker
  cat <<EOF > "$AUTOMATION_DIR/${NOTEBOOK_BASE}.sh"
#!/bin/bash

jupyter nbconvert --to python "/root/${NOTEBOOK_NAME}"
sed -i "s/Your_HuggingFace_Token/$TOKEN_KEY/g" "/root/${NOTEBOOK_BASE}.py"
ipython3 "/root/${NOTEBOOK_BASE}.py" 2>&1 | tee "/root/${NOTEBOOK_BASE}.out"

if [ \$? -ne 0 ]; then
    echo "TEST FAILED" >> "/root/${NOTEBOOK_BASE}.out"
else
    echo "TEST PASSED" >> "/root/${NOTEBOOK_BASE}.out"
fi
EOF

  chmod +x "$AUTOMATION_DIR/${NOTEBOOK_BASE}.sh"

  # Run Docker
  docker run --name "automation_${NOTEBOOK_BASE}" \
    --runtime=habana \
    -e HABANA_VISIBLE_DEVICES=all \
    -e OMPI_MCA_btl_vader_single_copy_mechanism=none \
    -v "${AUTOMATION_DIR}:/root" \
    --cap-add=sys_nice \
    --net=host \
    --ipc=host \
    --user root:root \
    sapdai/gaudi-tutorial-env:latest "/root/${NOTEBOOK_BASE}.sh"

  if [ $? -ne 0 ] ; then
    echo "Error executing ${NOTEBOOK_BASE}" | tee -a "$LOG_FILE"
  else
    echo "Successfully executed ${NOTEBOOK_BASE}" | tee -a "$LOG_FILE"
  fi

  # Cleanup
  docker container rm "automation_${NOTEBOOK_BASE}"
  rm "$AUTOMATION_DIR/${NOTEBOOK_BASE}.sh"
  rm "$AUTOMATION_DIR/${NOTEBOOK_BASE}.py"

done

# Final cleanup
#rm -rf "$AUTOMATION_DIR"

echo "Script execution completed. Check the log file at: $LOG_FILE"
