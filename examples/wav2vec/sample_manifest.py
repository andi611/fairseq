# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ sample_manifest.py ]
#   Synopsis     [ samples from a .tsv file until the given time frame is met, and output a subset of the input file. ]
#   Author       [ Andy T. Liu (Andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


import os
import random

def sample_manifest(input_file, output_file, max_timesteps):
    
    # Check if the output file already exists
    if os.path.exists(output_file):
        print("Output file already exists. Aborting save.")
        return
    
    lines = []
    total_count = 0

    with open(input_file, 'r') as file:
        lines = file.readlines()

    root_directory = lines[0].strip()
    data_lines = lines[1:]

    # Randomly shuffle the data lines
    random.shuffle(data_lines)

    sampled_lines = []
    for line in data_lines:
        line = line.strip().split('\t')
        string_value = line[0]
        int_value = int(line[1])

        if total_count + int_value <= max_timesteps:
            sampled_lines.append([string_value, str(int_value)])
            total_count += int_value
        else:
            remaining_timesteps = max_timesteps - total_count
            if remaining_timesteps > 0:
                sampled_lines.append([string_value, str(remaining_timesteps)])
                total_count += remaining_timesteps
            break

    # Check if the directory exists, if not, create the directory
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_file, 'w') as file:
        file.write(root_directory + '\n')
        for line in sampled_lines:
            file.write('\t'.join(line) + '\n')

    # Print the lengths of input and output files
    print("Input file length:", len(data_lines))
    print("Output file length:", len(sampled_lines))

# Example usage
n_hours = 10
input_file = '/work/a129195789/manifest/100/train.tsv'
output_file = f'/work/a129195789/manifest/{n_hours}/train.tsv'
max_timesteps = 16000 * 60 * 60 * n_hours

sample_manifest(input_file, output_file, max_timesteps)
