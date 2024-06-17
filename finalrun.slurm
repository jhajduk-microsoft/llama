#!/bin/bash

#SBATCH --job-name=llama2_inference
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=llama2_inference_%j.log

# Set Hugging Face API token as an environment variable
export HUGGINGFACE_TOKEN=""

# Ensure the token is available for the huggingface_hub library
export HF_HOME=$HOME/.cache/huggingface
mkdir -p $HF_HOME
echo $HUGGINGFACE_TOKEN > $HF_HOME/token

# Activate virtual environment
source llama2_env/bin/activate

#Upgrade pip
pip install --upgrade pip

#Install dependencies
sudo yum install -y libjpeg-devel
pip install transformers==4.28.0
pip install pillow torchvision torchaudio sentencepiece fairscale fire

# Define output file for the generated text
OUTPUT_FILE=/generated_text.txt

# Run the Python script
srun python finalrun.py --model_name /llama/model --prompt "Once upon a time" --max_length 100 --output_file $OUTPUT_FILE
