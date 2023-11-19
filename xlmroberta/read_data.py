import csv
import math


def clean_train_set_paragraph(df, len_df, split_num):
    """Removes from the train set the paragraphs that belong to the articles which will be part of
    the test set in the downstream task. 
    df - dataframe of training data
    split_num - the 10% split which will be used as the test set in the downstream task and should now be removed.
    """
    boundary_1 = math.floor(len_df * split_num * 0.1)
    boundary_2 = math.floor(len_df * (split_num + 1) * 0.1)
    print(boundary_1)
    print(boundary_2)
    train_data = []
    train_labels = []
    for index, row in df.iterrows():
        if row['nid'] < boundary_1 or row['nid'] > boundary_2:
            train_data.append(row['content'])
            train_labels.append(row['sentiment'])

    return train_data, train_labels


def read_croatian_data(data_path):
    cro_data = []
    cro_labels = []
    with open(data_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter="\t", quotechar='"')
        for i, line in enumerate(reader):
            if i == 0:
                continue
            cro_labels.append(line[0])
            text = line[1] + ". " + line[2]
            cro_data.append(text)

    return cro_data, cro_labels
