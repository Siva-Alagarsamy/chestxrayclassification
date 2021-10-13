# This script will convert the .bz2 train and test data file to the Keras dataset directory format
# The directory structure will be as follows
#    train/
#    ---pos/
#    ------review_1.txt
#    ------review_2.txt
#    ---neg/
#    ------review_1.txt
#    ------review_2.txt

import bz2
import os


def to_keras_format(txt_bz_file, output_path=".", ):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if not os.path.exists(f"{output_path}/pos"):
        os.makedirs(f"{output_path}/pos")

    if not os.path.exists(f"{output_path}/neg"):
        os.makedirs(f"{output_path}/neg")

    row_numbers = [0, 0, 0]
    label_names = ["", "neg", "pos"]

    with bz2.open(txt_bz_file, "rt", encoding='utf-8') as bz_file:
        for line in bz_file:
            # Label and review are separated by space
            label, review = line.split(' ', maxsplit=1)

            # label has a format __label__2  we just need the last number
            label = int(label[9:])

            row_numbers[label] += 1
            out_file_name = f"{output_path}/{label_names[label]}/review_{row_numbers[label]}.txt"
            out_file = open(out_file_name, "wt", encoding='utf-8')
            out_file.write(review)
            out_file.close()


def convert():
    to_keras_format("../data/test.ft.txt.bz2", "../data/test")
    to_keras_format("../data/train.ft.txt.bz2", "../data/train")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    convert()