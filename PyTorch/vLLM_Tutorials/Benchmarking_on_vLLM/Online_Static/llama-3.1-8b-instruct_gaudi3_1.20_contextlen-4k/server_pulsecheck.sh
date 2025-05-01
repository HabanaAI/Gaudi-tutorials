#!/bin/sh

runType=$1
target=$2

if [ -z $target ]; then
	target=localhost
fi
if [ ! -f /usr/bin/curl ]; then
	echo "Curl not found, installing"
	apk add curl
fi
# Allow a time gap for server to come up
echo -e "Waiting $server_up_sleep seconds for server to come up. This is a $runType check."
sleep $server_up_sleep

# Check if Server is up by using curl request
echo -e "\n$(date): Check if Server is up by sending curl request every 5 seconds"
    
payload="{ \"model\": \"${model}\", \"prompt\": \"${curl_query}\", \"max_tokens\": 128, \"temperature\": 0 }"
echo "Payload: $payload"

# Run a while loop inside the Docker container
DURATION=1500
START_TIME=$(date +%s)
while true; do
    # Execute the curl command and capture the HTTP status code
    HTTP_CODE=$(curl -s --noproxy '*' -o /dev/null -w '%{http_code}' http://${target}:8000/v1/completions \
        -H 'Content-Type: application/json' \
        -d "$payload" )

    # Check if the HTTP status code indicates success (200 OK)
    if [[ $HTTP_CODE -eq 200 ]]; then
        echo "Received a successful response with status code: $HTTP_CODE"
        if [ "$runType" = "initial" ]; then
			echo "Inital check complete, exiting script"
			break
        fi
    else
        echo "No successful response. HTTP status code: $HTTP_CODE. Retrying in 5 seconds..."
        sleep 5
    fi

    if [ "$runType" = "initial" ]; then
        # Check if the elapsed time has exceeded the specified duration
        CURRENT_TIME=$(date +%s)
        ELAPSED_TIME=$((CURRENT_TIME - START_TIME))
        if [ "$ELAPSED_TIME" -ge $DURATION ]; then
            echo "Elapsed time of $DURATION seconds reached. Exiting loop."
            break
        fi
    fi

done
