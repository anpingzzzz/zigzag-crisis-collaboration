import numpy as np
import os
import pickle
import itertools
from tqdm import tqdm
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# compute betti curve
def Get_betti_curve(dgms, iters, SWL, OL): 
    dgm_1dim = {}
    window_count = int((iters - SWL) / OL)
    for idx,dgm in enumerate(dgms):
        dgm_1dim[f'dim_{idx}'] = np.zeros(iters)
        points = [(int(point.birth), int(point.death) if point.death != float('inf') else window_count) for point in dgm] 
        for i in points:
            start = i[0] * OL
            end = min(i[1] * OL + SWL, iters)  
            dgm_1dim[f'dim_{idx}'][start:end] += 1
    return dgm_1dim

def get_betti_average(folder_path, num_files= 10):
    betti_0 = []
    betti_1 = []
    for i in range(1, num_files + 1):
        file_name = f"dgms_{i}.pkl"
        file_path = os.path.join(folder_path, file_name)
        if os.path.exists(file_path):
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
                dict_betti = Get_betti_curve(data, iters = 100, SWL=10, OL=5)
                betti_0.append(dict_betti['dim_0'])
                if 'dim_1' in dict_betti:
                    betti_1.append(dict_betti['dim_1'])
                else:
                    betti_1.append([0] * 100)
    average_betti_0 = np.mean(betti_0, axis=0)
    average_betti_1 = np.var(betti_1, axis=0)
    return average_betti_0, average_betti_1

# Compute averge system effectiveness from the simulation results
def get_effectiveness_average_and_variance(folder_path, num_files = 10):
    """
    folder_path (str): The path to the folder containing the files.
    num_files (int): The number of files to be read.
    """
    data_lists = []

    for i in range(1, num_files + 1):
        file_name = f"system_effectiveness_{i}.pkl"
        file_path = os.path.join(folder_path, file_name)
        if os.path.exists(file_path):
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
                if isinstance(data, list) and len(data) > 0: 
                    data_lists.append(data)

    if not data_lists:
        print("No valid data found.")
        return
    all_data = np.array(data_lists)
    average_data = np.mean(all_data, axis=0)
    variance_data = np.var(all_data, axis=0)
    return average_data, variance_data

if __name__ == "__main__":
    op = [10,20,30,40,50,60]
    agent_num = [600, 1200, 1800, 2400, 3000, 3600, 4200]
    print("[INFO] Computing average effectiveness and betti curve")
    avg_effectiveness = {
            f'op_{i}_agent_{j}': list(
                get_effectiveness_average_and_variance(
                    f'../results/simulation_results/{i}%/iters_100_poi_False_tasks_600_agents_{j}_org_percent_{str(i)[::-1]}_SWL_10_OL_5_Upref_equal_Opref_equal/'
                )
            )
            for i, j in itertools.product(op, agent_num)
        }
    avg_betti_curve = {
            f'op_{i}_agent_{j}': list(
                get_betti_average(
                    f'../results/simulation_results/{i}%/iters_100_poi_False_tasks_600_agents_{j}_org_percent_{str(i)[::-1]}_SWL_10_OL_5_Upref_equal_Opref_equal/'
                )
            )
            for i, j in tqdm(itertools.product(op, agent_num))
        }  
    print("[INFO] Saving the results")
    with open('../results/simulation_results/avg_betti_curve_replicate.pkl', 'wb') as file:
        pickle.dump(avg_betti_curve, file)
    with open('../results/simulation_results/avg_effectiveness_replicate.pkl', 'wb') as file:
        pickle.dump(avg_effectiveness, file)
    print("[INFO] Done")