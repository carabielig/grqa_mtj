#!/bin/bash

module load python
cd ~/grqa_mtj
source mtj_env/bin/activate

echo "Saving numpy arrays..."
python3 store_np_arrays.py bacteria /p/tmp/carabi/data/bacteria_images /p/tmp/carabi/data/bacteria_images_np

sbatch grqa_single_images /p/tmp/carabi/data/bacteria_images_np/deterministic_5.npy /p/tmp/carabi/bacteria_images 0.2 30 100 5

#for filename in /p/tmp/carabi/data/bacteria_images_np/*.npy; do
#	echo "Starting job for $filename ..."
#	sbatch grqa_single_images $filename /p/tmp/carabi/bacteria_images 0.2 30 100 5
#done
