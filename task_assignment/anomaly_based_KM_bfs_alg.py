import pickle
from pathlib import Path
import numpy as np
import time
from utils import KM, get_normal_worker_task_pairs, get_weights_by_weightmatrix_and_edges
import copy
import pandas as pd

dataset_list = ["MSL", "SMAP"]
hp_valid_time_list = [2, 4, 6, 8, 10]
prefix = ""
res_dataset_list, res_hp_valid_time_list, res_workers_list, res_tasks_list, res_weights_list, res_avg_weigths_list, res_ratio_task_list, res_ratio_worker_list, res_cpu_time_list, res_normal_worker_ratio_list, res_normal_task_ratio_list, res_normal_utility_ratio_list = [], [], [], [], [], [], [], [], [], [], [], []

if not Path("./results").exists():
    Path("./results/").mkdir(parents=True)

for dataset in dataset_list:
    for hp_valid_time in hp_valid_time_list:
        if dataset == "MSL":
            N = 2000
            n_workers_tasks_list = [[500, N], [1000, N], [1500, N], [2000, N], [N, 500], [N, 1000], [N, 1500]]
        else:
            N = 2000
            n_workers_tasks_list = [[500, N], [1000, N], [1500, N], [2000, N], [2500, N], [N, 500], [N, 1000], [N, 1500], [N, 2500]]
        if hp_valid_time != 10:
            n_workers_tasks_list = [[2000, 2000]]
        for n_workers, n_tasks in n_workers_tasks_list:
            workers_info_path = Path("./data/crowdsouring_data/{}/{}workers_info_x_{}_nworkers_{}_ntasks_{}.pkl".format(dataset, prefix, hp_valid_time, n_workers, n_tasks))
            tasks_info_path = Path("./data/crowdsouring_data/{}/{}tasks_info_x_{}_nworkers_{}_ntasks_{}.pkl".format(dataset, prefix, hp_valid_time, n_workers, n_tasks))
            weights_matrix_path = Path("./data/crowdsouring_data/{}/{}weights_matrix_x_{}_nworkers_{}_ntasks_{}.pkl".format(dataset, prefix, hp_valid_time, n_workers, n_tasks))

            _, _, workers_anomaly = pickle.load(workers_info_path.open("rb"))
            _, _, tasks_anomaly = pickle.load(tasks_info_path.open("rb"))
            weights_matrix, anomaly_weights_matrix = pickle.load(weights_matrix_path.open("rb"))

            start_time = time.time()
            if isinstance(anomaly_weights_matrix, list):
                px_list = []
                for i in range(len(anomaly_weights_matrix)):
                    _, px = KM(anomaly_weights_matrix[i])
                    px_list.append(copy.copy(px))
                px = None
            else:
                _, px = KM(anomaly_weights_matrix)

            end_time = time.time()
            cpu_time = end_time - start_time

            def convert_px_to_chose_edge(px, anomaly_weights_matrix):
                N = anomaly_weights_matrix.shape[0]
                chose_edge = []
                for i in range(1, N+1):
                    if anomaly_weights_matrix[i-1][px[i]-1] != 0:
                        chose_edge.append([i-1, px[i]-1])
                return chose_edge

            if isinstance(anomaly_weights_matrix, list):
                chose_edges = []
                for i in range(len(px_list)):
                    chose_edges.append(convert_px_to_chose_edge(px_list[i], anomaly_weights_matrix[i]))
            else:
                chose_edges = convert_px_to_chose_edge(px, anomaly_weights_matrix)

            # evaluate
            n_valid_task_worker_pairs = 0
            if isinstance(weights_matrix, list):
                res_list = []
                org_res_list = []
                n_normal_workers_list = []
                n_normal_tasks_list = []
                total_worker_task = 0
                for i in range(len(chose_edges)):
                    pure_chose_edge, n_normal_workers, n_normal_tasks = get_normal_worker_task_pairs(chose_edges[i], workers_anomaly[i].astype("int").tolist(), tasks_anomaly[i].astype("int").tolist())
                    res_list.append(get_weights_by_weightmatrix_and_edges(weights_matrix[i], pure_chose_edge))
                    org_res_list.append(get_weights_by_weightmatrix_and_edges(weights_matrix[i], chose_edges[i]))
                    n_valid_task_worker_pairs += len(pure_chose_edge)
                    n_normal_workers_list.append(n_normal_workers)
                    n_normal_tasks_list.append(n_normal_tasks)
                    total_worker_task += len(chose_edges[i])
                n_normal_tasks = sum(n_normal_tasks_list)
                n_normal_workers = sum(n_normal_workers_list)
                res = np.mean(res_list)
                org_res = np.mean(org_res_list)
                sum_res = sum(res_list)
                ratio_div = len(chose_edges)
            else:
                pure_chose_edge, n_normal_workers, n_normal_tasks = get_normal_worker_task_pairs(chose_edges, workers_anomaly.astype("int").tolist(), tasks_anomaly.astype("int").tolist())
                res = get_weights_by_weightmatrix_and_edges(weights_matrix, pure_chose_edge)
                org_res = get_weights_by_weightmatrix_and_edges(weights_matrix, chose_edges)
                sum_res = res
                n_valid_task_worker_pairs += len(pure_chose_edge)
                total_worker_task = len(chose_edges)
                ratio_div = 1

            normal_worker_ratio = n_normal_workers / total_worker_task
            normal_task_ratio = n_normal_tasks / total_worker_task
            normal_utility_ratio = res / org_res
            avg_weights = sum_res / n_valid_task_worker_pairs
            ratio_task = n_valid_task_worker_pairs / n_tasks / ratio_div
            ratio_worker = n_valid_task_worker_pairs / n_workers / ratio_div
            res_df = pd.DataFrame({"dataset": dataset, "Methods": "KM+outlier", "x": hp_valid_time, "workers": n_workers, "tasks": n_tasks, "weights": res,
                "avg_weights": avg_weights, "ratio(task)": ratio_task, "ratio(worker)": ratio_worker, "CPU time(s)": cpu_time, "normal_worker_ratio":
                normal_worker_ratio, "normal_task_ratio": normal_task_ratio, "normal_utility_ratio": normal_utility_ratio}, index=[0])
            res_df.to_csv("results/{}anomaly_based_KM_res_{}_x_{}_workers_{}_tasks_{}.csv".format(prefix, dataset, hp_valid_time, n_workers, n_tasks), index=False)
            print(res_df)
            res_dataset_list.append(dataset)
            res_hp_valid_time_list.append(hp_valid_time)
            res_workers_list.append(n_workers)
            res_tasks_list.append(n_tasks)
            res_weights_list.append(res)
            res_avg_weigths_list.append(avg_weights)
            res_ratio_task_list.append(ratio_task)
            res_ratio_worker_list.append(ratio_worker)
            res_cpu_time_list.append(cpu_time)
            res_normal_worker_ratio_list.append(normal_worker_ratio)
            res_normal_task_ratio_list.append(normal_task_ratio)
            res_normal_utility_ratio_list.append(normal_utility_ratio)
pd.DataFrame({"dataset": res_dataset_list, "Methods": "KM+outlier", "x":res_hp_valid_time_list, "workers": res_workers_list, "tasks": res_tasks_list,
    "weights": res_weights_list, "avg_weights": res_avg_weigths_list, "ratio(task)": res_ratio_task_list, "ratio(worker)": res_ratio_worker_list,
    "CPU time(s)": res_cpu_time_list, "normal_worker_ratio": res_normal_worker_ratio_list, "normal_task_ratio": res_normal_task_ratio_list, "normal_utility_ratio":
    res_normal_utility_ratio_list}).to_csv("results/{}anomaly_based_KM_res_CPUTime.csv".format(prefix), index=False)
