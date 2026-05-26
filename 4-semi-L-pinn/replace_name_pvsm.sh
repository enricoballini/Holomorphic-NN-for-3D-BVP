#!/bin/bash

CURRENT_FOLDER=$(basename "$PWD")
PVSM_FILE="./visualize_results.pvsm"
BASE_PATH="/home/loq/Desktop/Aarhus/3-laplacians_jax/cases_papkovich_neuber"

if [ ! -f "$PVSM_FILE" ]; then
    echo "Error: $PVSM_FILE not found."
    exit 1
fi

# Replace any existing case folder in the path with the current folder name
sed -i "s|${BASE_PATH}/[^/]*/|${BASE_PATH}/${CURRENT_FOLDER}/|g" "$PVSM_FILE"

echo "Replaced case folder with '${CURRENT_FOLDER}' in ${PVSM_FILE}"