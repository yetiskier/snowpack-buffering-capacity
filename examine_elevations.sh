#!/bin/bash

# This script automates running analysis on various elevation bands

for i in $(seq 900 100 2500);
do
echo "$((i)) to $((i+100))"
python elevationBinColdContentCalculate.py $((i)) $((i+100));
python swe_elevationBinSWECalculate.py $((i)) $((i+100));
python REHelevationBinREHCalculate.py $((i)) $((i+100));

done


