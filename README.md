###A pytorch script to achieve ParticleNet using DistributedDataParallel method

script_GPUonly and script_GPUorCPU are same but the latter will slower than GPUonly, so Please use script_GPUonly when sbatch the script.

to run the script:
just ./start_GPU.sh
check the settings and type a suffix for this training
then l means running locally(for test), s means sbatch

Please change the hyperparameters in the 

script_GPUonly/my_train_DDP.py
