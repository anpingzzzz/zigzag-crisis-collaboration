import random
import numpy as np
from shapely.geometry import Point, Polygon
import pandas as pd
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from itertools import combinations, chain
import dionysus as d
import utils.GetNewMethods as GetNewMethods
import argparse
import warnings
warnings.filterwarnings('ignore')
import pickle

class Street: 
    def __init__(self, _id, name, polygon_points, user_distribution, organizer_distribution):
        self.id = _id
        self.name = name
        self.polygon = Polygon(polygon_points)
        self.user_distribution = user_distribution
        self.organizer_distribution = organizer_distribution

    def generate_random_location(self):
        shenzhen_polygon = [(113.7678, 22.8531), (114.6674, 22.8531), (114.6674, 22.3999), (113.7678, 22.3999)]
        min_x, min_y, max_x, max_y = self.polygon.bounds
        while True:
            random_point = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
            if self.polygon.contains(random_point) and Polygon(shenzhen_polygon).contains(random_point):
                return random_point.x, random_point.y

class Task:
    def __init__(self, task_type, street, task_capacity):
        self.street = street
        self.task_type = task_type
        self.task_location = self.street.generate_random_location() # generate tasks in the selected street
        self.task_capacity = task_capacity
        self.participants = []
        self.organizer = None
              
    def is_full(self):
        return len(self.participants) >= self.task_capacity
    
    def compute_effectiveness(self):
        team_size = len(self.participants)
        team_cohesion = self.compute_team_cohesion()
        team_familiarity = self.compute_team_familiarity()
        effectiveness = team_size * team_cohesion * team_familiarity 
        return effectiveness

    def compute_team_cohesion(self):
        if len(self.participants) == 0:
            return 0
        team_cohesion = 0
        for i in range(len(self.participants)):
            for j in range(i + 1, len(self.participants)):
                if self.participants[j] in self.participants[i].history_collaboration:
                    team_cohesion+=1                
        return (team_cohesion / (len(self.participants) ** 2)) + 0.01

    def compute_team_familiarity(self):
        if len(self.participants) == 0:
            return 0
        total_experience = 0
        for agent in self.participants:
            total_experience += agent.experience[self.task_type]
        return total_experience / len(self.participants)

    def step(self):
        for u in self.participants:
            u.history_collaboration.update(self.participants)
        self.participants = []
        self.organizer = None
        if random.random() < 0.2:
            self.street = random.choice(list(model.streets.values()))
        if random.random() < 0.5:
            self.task_location = self.street.generate_random_location()
            self.task_capacity = random.randint(4, 7)


class Volunteer:
    id_counter = 0
    def __init__(self, home_street, streets_poi_dict, is_organizer, user_experience_data, v1, v2, v3, v4, v5):
        self.id = Volunteer.id_counter
        Volunteer.id_counter += 1
        self.home_street = home_street
        self.streets_poi_dict = streets_poi_dict
        self.home_location = home_street.generate_random_location()
        self.is_organizer = is_organizer
        self.history_collaboration = set()
        self.task_affinity = {
            "pandemic_tasks": 0.43,
            "business and school reopening": 0.02,
            "public outreach": 0.15,
            "transportation tasks": 0.08,
            "environmental_tasks": 0.13,
            "community service": 0.19
        }
        self.collaborative_propensity = 0
        self.experience = self.initialize_experience(user_experience_data)
        self.v1, self.v2, self.v3, self.v4, self.v5 = v1, v2, v3, v4, v5
        self.history = []

    def initialize_experience(self, user_experience_data):
        random_user = random.choice(list(user_experience_data.values()))
        experience = {
            "pandemic_tasks": random_user.get("pandemic_tasks", 0) * 0.01,
            "business and school reopening": random_user.get("business and school reopening", 0)* 0.01,
            "public outreach": random_user.get("public outreach", 0)* 0.01,
            "transportation tasks": random_user.get("transportation tasks", 0)* 0.01,
            "environmental_tasks": random_user.get("environmental_tasks", 0)* 0.01,
            "community service": random_user.get("community service", 0)* 0.01
        }
        return experience        

    def choose_task_as_organizer(self, tasks):
        available_tasks = [t for t in tasks if t.organizer is None]
        probabilities = []
        for task in available_tasks:
            distance = np.sqrt((task.task_location[0] - self.home_location[0])**2 + (task.task_location[1] - self.home_location[1])**2)
            task_affinity = self.task_affinity[task.task_type]
            if task.street == self.home_street: #agents may select other streets' tasks under a probability of 0.14 (sampled from real-world data)
                prob = self.v4 * (1/(1+distance)) + self.v5 * task_affinity 
            else:
                prob = (self.v4 * (1/distance) + self.v5 * task_affinity) * 0.14
            probabilities.append(prob)
        total_prob = sum(probabilities)
        if total_prob > 0:
            probabilities = [p / total_prob for p in probabilities]
        if available_tasks:
            chosen_task = random.choices(available_tasks, weights=probabilities, k=1)[0]
        else:
            chosen_task = None
        return chosen_task
    
    def choose_task_as_user(self, tasks):
        if random.random() < 0.2: # 20% of the time, the user will not choose any task
            return None
        available_tasks = [t for t in tasks if t.organizer is not None]
        probabilities = []
        for task in available_tasks:
            distance = np.sqrt((task.task_location[0] - self.home_location[0])**2 + (task.task_location[1] - self.home_location[1])**2)
            task_affinity = self.task_affinity[task.task_type]
            collaborative_propensity = np.sum([1 if u in self.history_collaboration else 0 for u in task.participants])
            if len(task.participants) > 1:  
                normalized_collaboration = collaborative_propensity / (len(task.participants) - 1)
            else:
                normalized_collaboration = 0  
            if task.street == self.home_street: #agents may select other streets' tasks under a probability of 0.14 (sampled from real-world data)
                prob = self.v1 * (1/(1+distance)) + self.v2 * task_affinity + self.v3 * normalized_collaboration 
            else:
                prob = (self.v1 * (1/(1+distance)) + self.v2 * task_affinity + self.v3 * normalized_collaboration) * 0.14 
            probabilities.append(prob)
        total_prob = sum(probabilities)
        if total_prob > 0:
            probabilities = [p / total_prob for p in probabilities]
        if available_tasks:
            chosen_task = random.choices(available_tasks, weights=probabilities, k=1)[0]
        else:
            chosen_task = None
        return chosen_task

    def step(self, tasks):
        # Implement the agent's step function
        task_info = None
        if self.is_organizer:
            chosen_task = self.choose_task_as_organizer(tasks)
            if chosen_task and not chosen_task.is_full():
                if chosen_task.organizer is None:
                    chosen_task.organizer = self    
                    chosen_task.participants.append(self)
                    task_info = (chosen_task.task_type, chosen_task.street.name, chosen_task.task_location)
        else:
            chosen_task = self.choose_task_as_user(tasks)
            if chosen_task and not chosen_task.is_full():
                chosen_task.participants.append(self)
                task_info = (chosen_task.task_type, chosen_task.street.name, chosen_task.task_location)
        
        self.history.append({
            'agent_id':self.id,
            'timestep': model.current_step,
            'role': 'organizer' if self.is_organizer else 'volunteer',
            'task_info': task_info,
            'home_location': self.home_location,
            'home_street': self.home_street.name
        })
class Scheduler:
    def __init__(self, strategy='random') -> None:
        self.collection = []
        self.strategy = strategy

    def step(self, *args, **kwargs):
        if self.strategy == 'random':
            random.shuffle(self.collection)
        for obj in self.collection:
            obj.step(*args, **kwargs)

class CollaborationModel:
    def __init__(self, total_agents_num, total_tasks_num, organizer_percentage, v1, v2, v3, v4, v5, task_distribution_df):
        self.streets, self.streets_poi_dict = self.read_street_data()
        self.organizer_schedule = Scheduler()
        self.participant_schedule = Scheduler()
        self.task_schedule = Scheduler()
        self.total_tasks_num = total_tasks_num
        self.total_agents_num = total_agents_num
        self.organizer_percentage = organizer_percentage
        self.tasks = self.generate_tasks(task_distribution_df)
        self.user_experience_data = self.read_user_experience_data()
        self.volunteers = self.generate_volunteers(total_agents_num, organizer_percentage, v1, v2, v3, v4, v5)
        self.current_step = 0
        self.is_organizer_turn = True

        self.datacollector = []
        self.removed_volunteer_data = []
        
        self.task_schedule.collection.extend(self.tasks)
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3
        self.v4 = v4
        self.v5 = v5

        
    def read_user_experience_data(self):
        with open('./data/user_experience.json', 'r') as file:
            return json.load(file)
    
    def step(self):
        if self.current_step >= 40:
            if self.current_step % 40 == 0:
                self.add_agents(int(self.total_agents_num * (1 - self.organizer_percentage)) * 2)
            elif self.current_step % 40 == 10:
                self.remove_agents(int(self.total_agents_num * (1 - self.organizer_percentage)) * 2)     
        self.organizer_schedule.step(self.tasks)
        self.participant_schedule.step(self.tasks)
        self.datacollector.append(np.mean([task.compute_effectiveness() for task in self.tasks]))
        self.current_step += 1
        self.task_schedule.step()
    
    def collect_volunteer_data(self):
        volunteer_data = []
        for volunteer in self.volunteers:
            volunteer_data.extend(volunteer.history)
        volunteer_data.extend(self.removed_volunteer_data)
        return volunteer_data

    def read_street_data(self):
        st_polygon = pd.read_csv('./data/street_polygon.csv')
        streets_data = {}
        for _id, data in st_polygon.iterrows():
            street_name = data['street_name']
            corr = data['street_polygon'].split(',')
            corr = [(float(c.split(' ')[0]), float(c.split(' ')[1])) for c in corr]
            user_distribution = data['user_distribution']
            organizer_distribution = data['organizer_distribution']
            streets_data[_id] = Street(_id, street_name, corr, user_distribution, organizer_distribution)
        st_poi = pd.read_csv('./data/street_poi.csv')
        streets_poi = {}
        for _, data in st_poi.iterrows():
            street_name = data['Unnamed: 0']
            streets_poi[street_name] = data.tolist()[1:]
        return streets_data, streets_poi
 
    def generate_tasks(self, task_distribution_df):
        tasks = []
        total_task_distribution = task_distribution_df.sum().sum()  
        for street in self.streets.values():
            if street.name in task_distribution_df.columns:
                tasks_dist = task_distribution_df[street.name].to_dict()
                num_tasks = int((task_distribution_df[street.name].sum() / total_task_distribution) * self.total_tasks_num)
                for _ in range(num_tasks):
                    task_type = np.random.choice(list(tasks_dist.keys()), p=list(tasks_dist.values()))
                    task_capacity = random.randint(4, 7)
                    tasks.append(Task(task_type, street, task_capacity))
        return tasks
    
    
    def generate_volunteers(self, total_agents_num, organizer_percentage, v1, v2, v3, v4, v5):
        volunteers = []
        for street in self.streets.values():
            num_agents = int((street.user_distribution) * total_agents_num)
            num_organizers = int(num_agents * organizer_percentage)
            if num_organizers == 0:
                num_organizers += 1
                num_agents -=1
            num_users = num_agents - num_organizers

            for _ in range(num_organizers):
                _volunteer = Volunteer(street, self.streets_poi_dict, True, self.user_experience_data, v1, v2, v3, v4, v5)
                volunteers.append(_volunteer)
                self.organizer_schedule.collection.append(_volunteer)

            for _ in range(num_users):
                _volunteer = Volunteer(street, self.streets_poi_dict, False, self.user_experience_data, v1, v2, v3, v4, v5)
                volunteers.append(_volunteer)
                self.participant_schedule.collection.append(_volunteer)
                
                
        return volunteers
    
    # #simulate the dynamic of agents
    def add_agents(self, num_agents):
        new_volunteers = []
        street_keys = list(self.streets.keys())
        for _ in range(num_agents):
            random_street_key = random.choice(street_keys)
            street = self.streets[random_street_key]
            is_organizer = False
            _volunteer = Volunteer(street, self.streets_poi_dict, is_organizer, self.user_experience_data, self.v1, self.v2, self.v3, self.v4, self.v5)
            new_volunteers.append(_volunteer)
            self.participant_schedule.collection.append(_volunteer)
        self.volunteers.extend(new_volunteers)
        
    def remove_agents(self, num_agents):
        volunteers_to_remove = [v for v in self.volunteers if not v.is_organizer]
        volunteers_to_remove = random.sample(volunteers_to_remove, min(num_agents, len(volunteers_to_remove)))
        for volunteer in volunteers_to_remove:
            self.removed_volunteer_data.extend(volunteer.history)
            self.volunteers.remove(volunteer)
            self.participant_schedule.collection.remove(volunteer)        
                 
class ZigzagPersistence:
    def __init__(self, df, date_range, SWL, OL, max_tuple_len = 3, overlapping_threshold = 0.1):
        self.df = df
        self.date_range = date_range
        self.SWL = SWL
        self.OL = OL
        self.max_tuple_len = max_tuple_len
        self.overlapping_threshold = overlapping_threshold
        self.sc_list = None
        self.times = None
        self.dgms = None

    def construct_graph(self, start, end):
        return self.df.loc[start:end].groupby(['organizer_id'], sort=True)[["agent_id"]].agg({"agent_id": lambda x: tuple(set(x))}).iloc[:,0].to_dict()

    def get_overlapping_terms(self, d):
        all_keys = set(d.keys())
        overlaps = {}

        for k_len in range(2, self.max_tuple_len + 1):
            for key_tuple in combinations(all_keys, k_len):
                overlap = set(d[key_tuple[0]])
                for key in key_tuple[1:]:
                    overlap = overlap.intersection(set(d[key]))

                if overlap:
                    key = tuple(sorted(key_tuple))
                    if key in overlaps:
                        overlaps[key] = list(set(overlaps[key]) | overlap)
                    else:
                        overlaps[key] = list(overlap)

        overlapping_percentages = {}
        for key in overlaps:
            terms = overlaps[key]
            union = set()
            for k in key:
                union |= set(d[k])
            overlapping_percentages[key] = float(len(terms)) / len(union)

        filtered_results = {}
        for key in overlaps:
            if overlapping_percentages[key] > self.overlapping_threshold:
                filtered_results[key] = {"overlapping_terms": overlaps[key], "overlapping_percentage": overlapping_percentages[key]}
        return filtered_results

    def consecutive_ranges(self, numbers, max_time):
        ranges = []
        start, end = None, None

        for num in numbers:
            if start is None:
                start = num
                end = num
            elif num == end + 1:
                end = num
            else:
                ranges.append(start)
                if end != max_time:
                    ranges.append(end + 1)
                start = num
                end = num
        if start is not None:
            ranges.append(start)
            if end != max_time:
                ranges.append(end + 1)

        return ranges

    def get_SCs_existence(self):
        res = {}
        print("[INFO] Construct graphs and compute zigzag persistence")
        for idx in tqdm(range(0, len(self.date_range) - self.SWL, self.OL)):
            dict_1 = self.construct_graph(self.date_range[idx], self.date_range[idx + self.SWL - 1])
            result = self.get_overlapping_terms(dict_1)
            vertex_list = []
            for key in result.keys():
                for val in key:
                    vertex_list.append(val)
            a = list(set(vertex_list))
            a.extend(list(result.keys()))
            for i in a:
                res.setdefault(i, []).append(idx // self.OL)
            res = dict(sorted(res.items(), key=lambda x: len(str(x[0])), reverse=True))
        sc_list = list(map(lambda x: [x] if isinstance(x, int) else list(x), res))
        window_counts = int((len(self.date_range) - self.SWL) / self.OL)
        times = [self.consecutive_ranges(i, max_time=window_counts) for i in res.values()]
        self.sc_list = sc_list
        self.times = times
        print("[INFO] ZIGZAG persistence computation completed!")
        return sc_list, times

    def compute_dgms(self):
        if self.sc_list is None or self.times is None:
            raise ValueError("sc_list and times must be computed first using get_SCs_existence.")
        f = d.Filtration(self.sc_list)
        _, dgms, _ = d.zigzag_homology_persistence(f, self.times)
        self.dgms = dgms
        return dgms

    def pd_plot(self, if_save=False, if_show=True):
        '''
        Plot persistence diagrams for each dimension separately.
        '''
        range_max = int((len(self.date_range) - self.SWL) / self.OL)
        if self.dgms is None:
            raise ValueError("dgms must be computed first using compute_dgms.")
        
        color_list = ['#A9A9A9', '#FF6347', '#4169E1', '#d22027']
        
        for dim in range(len(self.dgms)):
            plt.figure(figsize=(5, 5))  # New figure for each dimension
            x = [i.birth for i in self.dgms[dim]]
            y = [i.death for i in self.dgms[dim]]
            inf_indices = [index for index, value in enumerate(y) if value == np.inf]

            # Replace np.inf with range_max for plottin
            y = [range_max if item == np.inf else item for item in y]
            
            plt.scatter(x, y, c=color_list[dim], alpha=0.3, label=r'$\beta$' + f'{dim}')
            
            x_diag = list(range(range_max + 1))
            plt.plot(x_diag, x_diag, '-', c='black', linewidth=2)
            
            # Add a horizontal line at range_max position to represent np.inf
            if inf_indices:
                plt.hlines(range_max, xmin=0, xmax=range_max, colors='grey', linestyles='dashed', alpha=0.7)

            plt.title(f'Persistence Diagram Dimension {dim}')
            plt.legend()

            if if_save:
                plt.savefig(f'per_diagram_dim_{dim}.pdf', bbox_inches='tight')
            
            if if_show:
                plt.show()

    def GetPersStats(self, barcode, app=False):
        '''
        Modified from paper 10.1109/TPAMI.2023.3308391
        Ali, Dashti, et al. "A survey of vectorization methods in topological data analysis." 
        IEEE Transactions on Pattern Analysis and Machine Intelligence (2023).
        '''
        if np.size(barcode) > 0:
            bc_av0, bc_av1 = np.mean(barcode, axis=0)
            bc_std0, bc_std1 = np.std(barcode, axis=0)
            bc_med0, bc_med1 = np.median(barcode, axis=0)
            bc_iqr0, bc_iqr1 = np.subtract(*np.percentile(barcode, [75, 25], axis=0))
            bc_r0, bc_r1 = np.max(barcode, axis=0) - np.min(barcode, axis=0)
            bc_p10_0, bc_p10_1 = np.percentile(barcode, 10, axis=0)
            bc_p25_0, bc_p25_1 = np.percentile(barcode, 25, axis=0)
            bc_p75_0, bc_p75_1 = np.percentile(barcode, 75, axis=0)
            bc_p90_0, bc_p90_1 = np.percentile(barcode, 90, axis=0)

            avg_barcodes = (barcode[:,1] + barcode[:,0]) / 2
            bc_av_av = np.mean(avg_barcodes)
            bc_std_av = np.std(avg_barcodes)
            bc_med_av = np.median(avg_barcodes)
            bc_iqr_av = np.subtract(*np.percentile(avg_barcodes, [75, 25]))
            bc_r_av = np.max(avg_barcodes) - np.min(avg_barcodes)
            bc_p10_av = np.percentile(avg_barcodes, 10)
            bc_p25_av = np.percentile(avg_barcodes, 25)
            bc_p75_av = np.percentile(avg_barcodes, 75)
            bc_p90_av = np.percentile(avg_barcodes, 90)

            diff_barcode = np.abs(np.subtract([i[1] for i in barcode], [i[0] for i in barcode]))
            bc_lengthAverage = np.mean(diff_barcode)
            bc_lengthSTD = np.std(diff_barcode)
            bc_lengthMedian = np.median(diff_barcode)
            bc_lengthIQR = np.subtract(*np.percentile(diff_barcode, [75, 25]))
            bc_lengthR = np.max(diff_barcode) - np.min(diff_barcode)
            bc_lengthp10 = np.percentile(diff_barcode, 10)
            bc_lengthp25 = np.percentile(diff_barcode, 25)
            bc_lengthp75 = np.percentile(diff_barcode, 75)
            bc_lengthp90 = np.percentile(diff_barcode, 90)

            bc_count = len([i for i in diff_barcode if i > 10])
            ent = GetNewMethods.Entropy()
            bc_ent = ent.fit_transform([barcode])

            bar_stats = np.array([
                bc_av0, bc_av1, bc_std0, bc_std1, bc_med0, bc_med1,
                bc_iqr0, bc_iqr1, bc_r0, bc_r1, bc_p10_0, bc_p10_1,
                bc_p25_0, bc_p25_1, bc_p75_0, bc_p75_1, bc_p90_0,
                bc_p90_1,
                bc_av_av, bc_std_av, bc_med_av, bc_iqr_av, bc_r_av, bc_p10_av,
                bc_p25_av, bc_p75_av, bc_p90_av, bc_lengthAverage, bc_lengthSTD,
                bc_lengthMedian, bc_lengthIQR, bc_lengthR, bc_lengthp10,
                bc_lengthp25, bc_lengthp75, bc_lengthp90,
                bc_count,
                bc_ent[0][0]
            ])

            if app:
                bar_stats = np.array([
                    bc_av0, bc_std0, bc_med0, bc_iqr0, bc_r0, bc_p10_0, bc_p25_0, bc_p75_0, bc_p90_0,
                    bc_av1, bc_std1, bc_med1, bc_iqr1, bc_r1, bc_p10_1, bc_p25_1, bc_p75_1, bc_p90_1,
                    bc_av_av, bc_std_av, bc_med_av, bc_iqr_av, bc_r_av, bc_p10_av, bc_p25_av, bc_p75_av, bc_p90_av,
                    bc_lengthAverage, bc_lengthSTD, bc_lengthMedian, bc_lengthIQR, bc_lengthR, bc_lengthp10, bc_lengthp25, bc_lengthp75, bc_lengthp90,
                    bc_count,
                    bc_ent[0][0]
                ])
        else:
            bar_stats = np.array([
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0,
                0,
                0
            ])

        bar_stats[~np.isfinite(bar_stats)] = 0
        return bar_stats

    def vectorize_dgm(self):
        if self.dgms is None:
            raise ValueError("dgms must be computed first using compute_dgms.")
        def handle_infinity(value):
            return int(value) if not np.isinf(value) else np.iinfo(np.int32).max
        return list(chain.from_iterable([self.GetPersStats(np.array([[handle_infinity(point.birth) for point in dgm], [handle_infinity(point.death) for point in dgm]]).T) for dgm in self.dgms]))
    

def get_agent_info(model):
    volunteer_data = model.collect_volunteer_data()
    df = pd.DataFrame(volunteer_data) 
    df = df.sort_values(by='timestep').set_index('timestep').dropna(subset=['task_info'])
    df_user = df[df.role == 'volunteer']
    df_organizer = df[df.role == 'organizer']
    organizer_list = []
    print("[INFO] Get agents' collaboration details from the simulation!")
    for timestep in tqdm(range(args.iters)):
        for task in df_user.loc[timestep].task_info:
            organizer_list.append(int(df_organizer.loc[timestep][df_organizer.loc[timestep].task_info == task].agent_id))
    print("[INFO] Collaboration details get!")
    df_user.loc[:,'organizer_id'] = organizer_list
    return df_user

   
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Run CollaborationModel simulation with specified parameters.')
    parser.add_argument('--iters', type=int, default=100, help='Number of iterations for the simulation')
    parser.add_argument('--consider_POI', type=bool, default=False, help='Whether to consider POI in task type distributions')
    parser.add_argument('--total_tasks_num', type=int, default=600, help='Total number of tasks in the simulation')
    parser.add_argument('--total_agents_num', type=int, default=1800, help='Total number of agents in the simulation')
    parser.add_argument('--organizer_percentage', type=float, default=0.3, help='Organizer percentage')
    parser.add_argument('--v1', type=float, default=1, help='Distance preference for volunteers')
    parser.add_argument('--v2', type=float, default=1, help='Task preference/affinity for volunteers')
    parser.add_argument('--v3', type=float, default=1, help='Collaboration propensity for volunteers')
    parser.add_argument('--v4', type=float, default=1, help='Distance preference for task organizers')
    parser.add_argument('--v5', type=float, default=1, help='Task preference/affinity for task organizers')
    parser.add_argument('--SWL', type=int, default=10, help='Sliding window length for constructing simplicial complexes')
    parser.add_argument('--OL', type=int, default=5, help='Overlapping days for constructing simplicial complexes ')

    args = parser.parse_args()

    task_distribution_df = pd.read_csv('./data/street_task_distribution_from_poi.csv', index_col=0).T if args.consider_POI else pd.read_csv('./data/street_tast_type_all_same.csv', index_col=0)

    model = CollaborationModel(
        total_tasks_num=args.total_tasks_num, 
        total_agents_num=args.total_agents_num,
        organizer_percentage=args.organizer_percentage,
        v1=args.v1,
        v2=args.v2,
        v3=args.v3,
        v4=args.v4,
        v5=args.v5,
        task_distribution_df=task_distribution_df
    ) 

    print("[INFO] Start to run the simulation!")
    for _ in tqdm(range(args.iters)):     
        model.step()
    print("[INFO] The simulation is completed!")

    system_effectiveness = model.datacollector
    agent_info = get_agent_info(model)
    zigzag_persistence = ZigzagPersistence(agent_info, range(args.iters), SWL=args.SWL, OL=args.OL, overlapping_threshold=0.05)
    sc_list, times = zigzag_persistence.get_SCs_existence()
    dgms = zigzag_persistence.compute_dgms()
    with open('system_effectiveness.pkl', 'wb') as f:
        pickle.dump(system_effectiveness, f)

    with open('dgms.pkl', 'wb') as f:
        pickle.dump(dgms, f)

    with open('agent_info.pkl', 'wb') as f:
        pickle.dump(agent_info, f)
    print("[INFO] Results have been saved!")
    
    


