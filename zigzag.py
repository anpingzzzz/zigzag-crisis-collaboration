import numpy as np
import pickle
import pandas as pd
from itertools import combinations
import dionysus as d
from tqdm import tqdm   
import argparse


class ZigzagPersistence:
    def __init__(self, SWL, OL, T, overlapping_threshold=0.1, max_tuple_len=3, start='2020/02/14', end='2023/09/30'):
        self.SWL = SWL
        self.OL = OL
        self.T = T
        self.overlapping_threshold = overlapping_threshold
        self.max_tuple_len = max_tuple_len
        self.start = start
        self.end = end
        self.date_range = pd.date_range(start, end)
        self.sc_list = None
        self.times = None

    def construct_graph(self, df_input, start_date, end_date):
        df_sliced = df_input[start_date:end_date]
        organizer_grouped = df_sliced.groupby(['organizer_id'], sort=True)[["user_id"]].agg({"user_id": lambda x: tuple(set(x))})
        organizer_grouped['user_count'] = [len(i) for i in organizer_grouped['user_id']]
        df_filter = organizer_grouped.loc[organizer_grouped['user_count'] > self.T]
        return df_filter.drop('user_count', axis=1).iloc[:, 0].to_dict()

    def get_overlapping_terms(self, d, max_tuple_len, threshold=0.0):
        all_keys = set(d.keys())
        overlaps = {}

        for k_len in range(2, max_tuple_len + 1):
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
        for key, terms in overlaps.items():
            union = set()
            for k in key:
                union |= set(d[k])
            overlapping_percentages[key] = float(len(terms)) / len(union)

        return {
            key: {
                "overlapping_terms": overlaps[key],
                "overlapping_percentage": overlapping_percentages[key],
            }
            for key in overlaps
            if overlapping_percentages[key] > threshold
        }

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

    def get_SCs_existence(self, df_input, save_path=None):
        res = {}
        print(f"[INFO] Computing lists of simplicial complexes from {len(list(range(0, len(self.date_range) - self.SWL, self.OL)))} sliding windows...")
        for idx in tqdm(range(0, len(self.date_range) - self.SWL, self.OL)):
            dict_1 = self.construct_graph(df_input, str(self.date_range[idx])[:10], str(self.date_range[idx + self.SWL - 1])[:10])
            result = self.get_overlapping_terms(dict_1, self.max_tuple_len, threshold=self.overlapping_threshold)
            vertex_list = []
            for key in result.keys():
                for val in key:
                    vertex_list.append(val)
            a = list(set(vertex_list))
            a.extend(list(result.keys()))
            for i in a:
                res.setdefault(i, []).append(idx // self.OL)
            res = dict(sorted(res.items(), key=lambda x: len(str(x[0])), reverse=True))
        self.sc_list = list(map(lambda x: [x] if isinstance(x, int) else list(x), res))
        window_counts = int((len(self.date_range) - self.SWL) / self.OL)
        self.times = [self.consecutive_ranges(i, max_time=window_counts) for i in res.values()]
        if save_path:
            with open(f"sc_list_{save_path}.pkl", "wb") as f:
                pickle.dump(self.sc_list, f)
            with open(f"times_{save_path}.pkl", "wb") as f:
                pickle.dump(self.times, f)

    def compute_dgms(self):
        print("[INFO] Computing persistence diagrams...")
        if self.sc_list is None or self.times is None:
            raise ValueError("sc_list and times must be computed first using get_SCs_existence.")
        f = d.Filtration(self.sc_list)
        _, dgms, _ = d.zigzag_homology_persistence(f, self.times)
        return dgms

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run ZigzagPersistence analysis with specified parameters.')
    parser.add_argument('--input_file', type=str, default='./data/volunteer_acticities.parquet',
                        help='Path to the input CSV file')
    parser.add_argument('--SWL', type=int, default=14, help='Sliding Window Length for ZigzagPersistence')
    parser.add_argument('--OL', type=int, default=7, help='Overlapping timesteps for ZigzagPersistence')
    parser.add_argument('--T', type=int, default=50, help='Minimum size for construct volunteer groups')
    parser.add_argument('--overlapping_threshold', type=float, default=0.1, help='Threshold for overlapping in SC detection')
    parser.add_argument('--end_date', type=str, default='2021/09/30', help='End date for the data')

    args = parser.parse_args()
    df_washed = pd.read_parquet(args.input_file).set_index('issued_time')
    df_input = df_washed.set_index(pd.to_datetime(df_washed.index)).drop_duplicates().sort_index()
    zzp = ZigzagPersistence(SWL=args.SWL, OL=args.OL, T=args.T, overlapping_threshold=args.overlapping_threshold, end=args.end_date)
    zzp.get_SCs_existence(df_input, save_path=f'SWL_{args.SWL}_OL_{args.OL}_Min_size_{args.T}_ol_{int(args.overlapping_threshold * 100)}')
    dgm = zzp.compute_dgms()
    print(dgm)
