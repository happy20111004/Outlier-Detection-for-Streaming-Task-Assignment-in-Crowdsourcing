set -ve
# Running the anomaly detection method proposed in our paper
cd ./SA-GAN
#python main.py --dataset="MSL"
#python main.py --dataset="SMAP"

# Running the task assignments
cd ../
cd ./task_assignment/
# The preprocess maybe cost more than half hour.
#python preprocess_MSL.py
#python preprocess_SMAP.py
python greedy_alg.py
python anomaly_based_greedy_alg.py
python KM_bfs_alg.py
python anomaly_based_KM_bfs_alg.py
