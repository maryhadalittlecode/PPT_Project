# PPT_Project

# ENV setups
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
conda --version
conda create -n canny_parallel python=3.10 -y
pip install -r requirements.txt


# How to run
python '/home/accts/xx242/workspace/project/PPT_Project/run.py' --data_dir data/ --output_dir outputs/ --method sequential_bfs

python '/home/accts/xx242/workspace/project/PPT_Project/run.py' --data_dir data/ --output_dir outputs/ --method frontier_parallel
