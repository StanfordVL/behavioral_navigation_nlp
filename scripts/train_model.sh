#!/bin/bash
#
#all commands that start with SBATCH contain commands that are just used by SLURM for scheduling
#################
#partition name
#SBATCH --partition=napoli-gpu
#################
#number of GPUs
#SBATCH --gres=gpu:1
#################
#set a job name
#SBATCH --job-name=basic-model
#################
#a file for job output, you can check job progress, append the job ID with %j to make it unique
#SBATCH --output=/cvgl/group/JR2.0/deep_nav_data/cluster_out/beh_nav_experiment1.%j.out
#################
# a file for errors from the job
#SBATCH --error=/cvgl/group/JR2.0/deep_nav_data/cluster_out/beh_nav_experiment1.%j.err
#################
#time you think you need; default is 2 hours
#format could be dd-hh:mm:ss, hh:mm:ss, mm:ss, or mm
#SBATCH --time=10:30:00
#################
# Quality of Service (QOS); think of it as sending your job into a special queue; --qos=long for with a max job length of 7 days.
# uncomment ##SBATCH --qos=long if you want your job to run longer than 48 hours, which is the default for normal partition,
# NOTE- in the hns partition the default max run time is 7 days , so you wont need to include qos, also change to normal partition
# since dev max run time is 2 hours.
##SBATCH --qos=long
# We are submitting to the dev partition, there are several on sherlock: normal, gpu, bigmem (jobs requiring >64Gigs RAM)
##SBATCH -p dev
#################
# --mem is memory per node; default is 4000 MB per CPU, remember to ask for enough mem to match your CPU request, since
# sherlock automatically allocates 4 Gigs of RAM/CPU, if you ask for 8 CPUs you will get 32 Gigs of RAM, so either
# leave --mem commented out or request >= to the RAM needed for your CPU request.  It will also accept mem. in units, ie "--mem=4G"
#SBATCH --mem=12000
# to request multiple threads/CPUs use the -c option, on Sherlock we use 1 thread/CPU, 16 CPUs on each normal compute node 4Gigs RAM per CPU.  Here we will request just 1.
#SBATCH -c 1
#################
# Have SLURM send you an email when the job ends or fails, careful, the email could end up in your clutter folder
# Also, if you submit hundreds of jobs at once you will get hundreds of emails.
#SBATCH --mail-type=END,FAIL # notifications for job done & fail
#################
#SBATCH --mail-user=ashwini1@stanford.edu

# list out some useful information
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES="=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "SLURM_JOB_NAME="$SLURM_JOB_NAME
echo "working directory = "$SLURM_SUBMIT_DIR

######time_stamp=$(date +%Y-%m-%d)
######mkdir -p "/cvgl/group/JR2.0/deep_nav_data/cluster_out/${time_stamp}"

#now run normal batch commands
# Begins in jr1_social_nav_experimental/jr2_deep_nav/cluster
umask 002
cd ..
git pull
source bnnlp/bin/activate
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:/usr/local/cuda-9.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export PATH=/usr/local/cuda-9.0/bin:/usr/local/cuda-9.1/bin${PATH:+:${PATH}}

if [[ $HOSTNAME == "napoli111" ]];
then
python codes/main.py --experiment_name=$SLURM_JOB_NAME\_$SLURM_JOBID --data_dir /scr/beh-nav-data
cp -R experiments/log_$SLURM_JOB_NAME\_$SLURM_JOBID /cvgl/group/JR2.0/deep_nav_data/cluster_out/logs ;
else
echo "Cannot find host!"
echo $HOSTNAME
exit 1
fi

# done
echo "Done"
exit 0
