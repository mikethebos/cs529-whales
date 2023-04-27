"""
@author Jack Ringer, Mike Adams
Date: 4/22/2023
Description:
Contains code to get baseline accuracy for non-ML methods
(using image duplicates / "new_whale" prediction)
"""

import pandas as pd


def generate_all_new_whale_submission(sample_sub_csv: str, save_pth: str):
    """
    Generate a submission where every entry is just "new_whale"
    :param sample_sub_csv: str, path to sample submission file
    :param save_pth: str, path to save new_whale submission to
    :return: None
    """
    df = pd.read_csv(sample_sub_csv)
    df["Id"] = "new_whale new_whale new_whale new_whale new_whale"
    df.to_csv(save_pth, index=False)


def main():
    generate_all_new_whale_submission("../data/sample_submission.csv", "../data/all_new_whale.csv")


if __name__ == "__main__":
    main()
