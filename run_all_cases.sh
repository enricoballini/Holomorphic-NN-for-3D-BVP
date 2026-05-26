#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

for case in \
  1-uniaxial \
  2-biaxial \
  3-shear \
  4-semi-L \
  4-semi-L-pinn \
  100-device-16 \
  100-device-32 \
  100-device-64-4 \
  100-device-64-8 \
  100-device-64-32 \
  100-device-64-default \
  100-device-128 \
  101-flow \
  101-flow-pinn
do
  echo "Running ${case}"
  (cd "${case}" && ./run.sh)
done
