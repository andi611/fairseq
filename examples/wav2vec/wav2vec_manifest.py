#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Data pre-processing: build vocabularies and binarize training data.
"""

import argparse
import glob
import os
import random

import soundfile


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root", metavar="DIR", help="root directory containing flac files to index"
    )
    parser.add_argument(
        "--dest", default=".", type=str, metavar="DIR", help="output directory"
    )
    parser.add_argument(
        "--ext", default="flac", type=str, metavar="EXT", help="extension to look for"
    )
    return parser

def is_train(file_path):
    if "train-clean-100" in file_path:
        return True
    elif "train-clean-360" in file_path:
        return True
    elif "train-other-500" in file_path:
        return True
    else:
        return False

def is_valid(file_path):
    if "dev-clean" in file_path:
        return True
    else:
        return False

def main(args):

    if not os.path.exists(args.dest):
        os.makedirs(args.dest)

    dir_path = os.path.realpath(args.root)
    search_path = os.path.join(dir_path, "**/*." + args.ext)

    valid_f = open(os.path.join(args.dest, "valid.tsv"), "w")

    with open(os.path.join(args.dest, "train.tsv"), "w") as train_f:
        print(dir_path, file=train_f)

        if valid_f is not None:
            print(dir_path, file=valid_f)

        for fname in glob.iglob(search_path, recursive=True):
            file_path = os.path.realpath(fname)

            if args.path_must_contain and args.path_must_contain not in file_path:
                continue

            frames = soundfile.info(fname).frames
            if is_train(file_path):
                dest = train_f
            elif is_valid(file_path):
                dest = valid_f
            else:
                continue
            print(
                "{}\t{}".format(os.path.relpath(file_path, dir_path), frames), file=dest
            )
    valid_f.close()


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
