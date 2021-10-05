import pickle
from pathlib import Path
import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler

start_time = time.time()
# hyperparamters
delta = 1e-10
N = 2500                                        # In SMAP, the length of time instance is set to 2500. SMAP: E-3 and E-6
anomaly_method = "2"
suffix = "npy"
prefix = ""
hp_valid_time_list = [2, 4, 6, 8, 10]

if not Path("./data/crowdsouring_data/SMAP").exists():
    Path("./data/crowdsouring_data/SMAP").mkdir(parents=True)

for hp_valid_time in hp_valid_time_list:
    n_workers_tasks_list = [[500, 2000], [1000, 2000], [1500, 2000], [2000, 2000], [2500, 2000], [2000, 500], [2000, 1000], [2000, 1500], [2000, 2500]]
    if hp_valid_time != 10:
        n_workers_tasks_list = [[2000, 2000]]
    for n_workers, n_tasks in n_workers_tasks_list:
        print("x: {}, n_workers: {}, n_tasks: {}".format(hp_valid_time, n_workers, n_tasks))
        workers_path = Path("./data/time_series_data/SA-point_score/pred/SMAP/E-3.{}".format(suffix))
        workers_anomaly_path = Path("./data/time_series_data/SA-point_score/label/SMAP/E-3.{}".format(suffix))
        tasks_path = Path("./data/time_series_data/SA-point_score/pred/SMAP/E-6.{}".format(suffix))
        tasks_anomaly_path = Path("./data/time_series_data/SA-point_score/label/SMAP/E-6.{}".format(suffix))
        save_workers_info_path = Path("./data/crowdsouring_data/SMAP/{}workers_info_x_{}_nworkers_{}_ntasks_{}.pkl".format(prefix, hp_valid_time, n_workers, n_tasks))
        save_tasks_info_path = Path("./data/crowdsouring_data/SMAP/{}tasks_info_x_{}_nworkers_{}_ntasks_{}.pkl".format(prefix, hp_valid_time, n_workers, n_tasks))
        save_weights_matrix_path = Path("./data/crowdsouring_data/SMAP/{}weights_matrix_x_{}_nworkers_{}_ntasks_{}.pkl".format(prefix,hp_valid_time, n_workers, n_tasks))

        # Suppose the gap between two workers or two tasks is one second.
        if suffix == "npy":
            workers = np.load(workers_path)
            workers_anomaly_label = np.load(workers_anomaly_path)
            tasks = np.load(tasks_path)
            tasks_anomaly_label = np.load(tasks_anomaly_path)
        else:
            workers = pickle.load(workers_path.open("rb"))
            workers_anomaly_label = pickle.load(workers_anomaly_path.open("rb"))
            tasks = pickle.load(tasks_path.open("rb"))
            tasks_anomaly_label = pickle.load(tasks_anomaly_path.open("rb"))

        # get three worker-task group
        workers = [workers[:N], workers[N:2*N], workers[2*N:3*N]]
        workers_anomaly = [workers_anomaly_label[:N], workers_anomaly_label[N:2*N], workers_anomaly_label[2*N:3*N]]
        tasks = [tasks[:N], tasks[N:2*N], tasks[2*N:3*N]]
        tasks_anomaly = [tasks_anomaly_label[:N], tasks_anomaly_label[N:2*N], tasks_anomaly_label[2*N:3*N]]

        # generating valid time of workers and tasks
        def generate_valid_time_by_arrival_time(arrival_time):
            valid_time_list = []
            for at in arrival_time:
                valid_time_list.append(float(np.random.uniform(at+delta, at+hp_valid_time+delta)))
            return valid_time_list

        workers_arrival_time = [list(range(N))] * 3
        tasks_arrival_time = [list(range(N))] * 3
        workers_valid_time = [generate_valid_time_by_arrival_time(workers_arrival_time[ti]) for ti in range(len(workers))]
        tasks_valid_time = [generate_valid_time_by_arrival_time(tasks_arrival_time[ti]) for ti in range(len(tasks))]

        if n_workers < N:
            print("random choice workers: {}".format(n_workers))
            choose_workers = np.random.choice(N, size=n_workers, replace=False)
        else:
            choose_workers = list(range(N))
        if n_tasks < N:
            print("random choice tasks: {}".format(n_tasks))
            choose_tasks = np.random.choice(N, size=n_tasks, replace=False)
        else:
            choose_tasks = list(range(N))

        # generating weight between workers and tasks
        # if there is no overlap between the valid_time of workers and tasks, there is no connection.
        weights_matrix = [np.zeros((N, N)) for _ in range(len(workers))]
        anomaly_weights_matrix = [np.zeros((N, N)) for _ in range(len(workers))]
        for i in range(len(weights_matrix)):
            for worker_id in range(N):
                if worker_id not in choose_workers:
                    continue
                ws, we = workers_arrival_time[i][worker_id], workers_valid_time[i][worker_id]
                for task_id in range(N):
                    if task_id not in choose_tasks:
                        continue
                    ts, te = tasks_arrival_time[i][task_id], tasks_valid_time[i][task_id]
                    # overlap
                    if (we-ws)+(te-ts) > max([ws,we,ts,te])-min([ws,we,ts,te]):
                        # original weights: U
                        weights_matrix[i][worker_id][task_id] = np.random.uniform(10-1e-10,20+1e-10)
                        # anomaly based weights: U * (1-max(w_anomaly, t_anomaly))
                        anomaly_weights_matrix[i][worker_id][task_id] = weights_matrix[i][worker_id][task_id] * \
                            (1-max(workers[i][worker_id],tasks[i][task_id]))

        pickle.dump([workers_arrival_time, workers_valid_time, workers_anomaly], save_workers_info_path.open("wb"))
        pickle.dump([tasks_arrival_time, tasks_valid_time, tasks_anomaly], save_tasks_info_path.open("wb"))
        pickle.dump([weights_matrix, anomaly_weights_matrix], save_weights_matrix_path.open("wb"))

        print("Cost time: ", time.time()-start_time)
