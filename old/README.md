# old

This directory contains obsolete scripts that were no longer needed.
Still, they were kept, because they might include interesting code fragments
that could be useful in the future. 

### cluster_individual_images

These scripts came from the idea that on the cluster each individual image could be processed
as an individual job. This was done by submitting the jobs through a shell file (`run_grqa_individual_images.sh`).
All data was stored as numpy arrays to get a consistent data type. Then the jobs were submitted.  

This attempt failed due to a limit of the number of jobs per user on the cluster.

### grqa

This directory contains intermediate versions of the line sampling that were later improved.

### multiprocessing

This script includes the attempt of parallelization on the CPU. This idea was abandoned due to
limited time and the promising results of parallelization on the GPU.
