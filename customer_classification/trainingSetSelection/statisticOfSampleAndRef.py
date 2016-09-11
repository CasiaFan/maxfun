import pandas as pd
import numpy as np
import re
import os

# file input directory
def statistic_machine_vs_human(int_dir, out_dir, marker_index, prefix_machine, prefix_human, prefix_result):
    # input: int_dir: directory containing machine and human mark files starting with prefix_machine and prefix_human mark respectively;
    # out_dir: output directory for result file starting with prefix_result mark. marker_index: from which columns containing manual marks
    # machine sample files
    machine_files = [file for file in os.listdir(int_dir) if re.match(prefix_machine+'[0-9]+.csv', file)]
    for machine_file in machine_files:
        # get enterprise_id of the input file
        sample = os.path.join(int_dir, machine_file)
        enterprise = re.findall(r'\d+', machine_file)
        human_file = int_dir + prefix_human + ''.join(enterprise) + ".csv"
        out_file = out_dir + prefix_result + ''.join(enterprise) + ".match.csv"

        df_ref = pd.read_csv(human_file)
        df_sample = pd.read_csv(sample)
        df_ref.index = df_ref['customer_id']
        df_sample.index = df_sample.customer_id
        df_marker_col = df_ref.ix[:, marker_index:]

        df_marker_col = df_marker_col.replace(['churn', 'potential_churn', 'active'], [3, 2, 1])
        df_ref['summary_marker'] = np.round(df_marker_col.mean(axis=1))
        df_ref['machine_marker'] = df_sample.mark.replace(['churn', 'potential_churn', 'active'], [3, 2, 1])
        # drop NA rows
        df_ref = df_ref.dropna()
        df_ref['match_or_not'] = (map(int, df_ref.summary_marker) == df_ref.machine_marker) * 1
        df_ref['past_match_or_not'] = (df_ref.marker.replace(['churn', 'potential_churn', 'active'], [3, 2, 1]) == map(int, df_ref.summary_marker)) * 1
        df_ref.to_csv(out_file)


int_dir = "C:/Users/fanzo/Desktop/validation_summary/"
out_dir = int_dir
prefix_machine = "test-"
prefix_human = "test_sample_from_"
prefix_result = "result_"
marker_index = 10
statistic_machine_vs_human(int_dir, out_dir, marker_index, prefix_machine, prefix_human, prefix_result)

