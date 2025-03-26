#!/bin/bash

if [ ! -x /usr/bin/envsubst ]; then
	echo "Need envsubst command needed but not founded"
	echo "Install with command 'apt install -y gettext and then re-run this script"
	exit -1
fi

cmdTemplate=server_cmd_template.txt
cmdPackaged=server_cmd_packaged.sh
cmdCurrent=server_cmd_current.sh

set -a

source docker_envfile.env
if [[ $HF_TOKEN == *"Here"* ]]; then
        echo "Could not find a token for HuggingFace. Cannot proceed without it!"
        echo "Add your HuggingFace token to docker_envfile.env"
        echo "For more details: https://huggingface.co/docs/hub/en/security-tokens"
	
        exit -1
fi

if [ ! -e ./hf_cache ];then
	echo "Creating ./hf_cache directory for storing cache."
	echo "If you have an existing HuggingFace Cache location and want to reuse it,"
	echo "create a link to it with:  ln -sf <existing_hf_path> ./hf_cache"
	mkdir hf_cache
fi

# Translate HABANA_VISIBLE_DEVICES to HABANA_VISIBLE_MODULES
TEMP_FILE=$(mktemp)
hl-smi -Q index,module_id -f csv > "$TEMP_FILE"
IFS=',' read -r -a devices <<< "$HABANA_VISIBLE_DEVICES"
modules=()

# Loop through each device index provided
for device_index in "${devices[@]}"; do
    device_index=$(echo "$device_index" | xargs)
    module_id=$(awk -F', ' -v idx="$device_index" '$1 == idx {print $2}' "$TEMP_FILE")
    modules+=("$module_id")
done

HABANA_VISIBLE_MODULES=$(IFS=, ; echo "${modules[*]}")
if grep -q HABANA_VISIBLE_MODULES docker_envfile.env; then
	sed -i "s/HABANA_VISIBLE_MODULES.*/HABANA_VISIBLE_MODULES=${HABANA_VISIBLE_MODULES}/g" docker_envfile.env
else
	echo "HABANA_VISIBLE_MODULES=$HABANA_VISIBLE_MODULES" >> docker_envfile.env
fi

# Clean up the temporary file
rm "$TEMP_FILE"

envsubst < $cmdTemplate > $cmdCurrent

shaCurrent=$(sha256sum $cmdCurrent)
shaPackaged=$(sha256sum $cmdPackaged)

if [ "$shaCurrent" != "$shaPackaged" ]; then
	echo "Input variables in docker_envfile have been modified! Creating new server_cmd.sh"
	ln -sf server_cmd_current.sh server_cmd.sh
else
	echo "No change in docker_envfile.env. Running with packaged server_cmd.sh"
	ln -sf server_cmd_packaged.sh server_cmd.sh
fi

chmod +x *.sh
if [ "$1" = "bg" ]; then
	docker compose up -d
else
	docker compose up
fi

