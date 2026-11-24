import argparse
from scripts import utils
import numpy as np
import pandas as pd
import os
import tqdm

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "image_type",
        help="The type of image that is processed, one of: test, bacteria, dtm, slp",
    )
    parser.add_argument(
        "image_path", help="Path to where the original images are stored"
    )
    parser.add_argument("out_path", help="Path to the output directory")

    args = parser.parse_args()

    if args.image_type == "test":
        test_images, epsilon_list, name_list = utils.collect_test_images(100)
    elif args.image_type == "bacteria":
        test_images, epsilon_list, name_list = utils.collect_bacteria_images(
            args.image_path
        )
    elif args.image_type == "dtm":
        test_images, epsilon_list, name_list = utils.collect_dtm(args.image_path)
    elif args.image_type == "slp":
        test_images, epsilon_list, name_list = utils.collect_slp(args.image_path)

    for i in tqdm.trange(len(test_images)):

        df_img = pd.DataFrame(
            {"epsilon": epsilon_list[i], "name": name_list[i]}, index=[0]
        )
        df_img.to_csv(os.path.join(args.out_path, name_list[i] + ".csv"), index=False)

        np.save(os.path.join(args.out_path, name_list[i] + ".npy"), test_images[i])
