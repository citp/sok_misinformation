#!/bin/bash

OUTPUT_FILE_ISOT="final_metrics_isot.txt"
OUTPUT_FILE_FAKES="final_metrics_fakes.txt"

OUTPUT_FILE_NYT="final_metrics_nyt.txt"
OUTPUT_FILE_REU="final_metrics_reu.txt"

seeds=(384 328 479 21 304 355 285 105 135 263 91 88 73 177 7 66 492 344 402 274 467 413 339 427 201 373 214 223 366 246)

for seed in "${seeds[@]}";
do
  echo "Running with random seed: $seed"
  # python CNN_RNN.py --test_source isot --rand_seed "$seed" >> $OUTPUT_FILE_ISOT &
  # python CNN_RNN.py --test_source fakes --rand_seed "$seed" >> $OUTPUT_FILE_FAKES &
  
  python CNN_revised.py --test_source nytimes --rand_seed "$seed" >> $OUTPUT_FILE_NYT &
  python CNN_revised.py --test_source reuters --rand_seed "$seed" >> $OUTPUT_FILE_REU &

  wait  

done