from read_data import clean_train_set_paragraph

import pandas as pd


def test_clean_train_set_paragraph():
    nid = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    content = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    sentiment = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    d = {'nid': nid, 'content': content, 'sentiment': sentiment}
    df = pd.DataFrame(data=d)
    train_data, train_labels = clean_train_set_paragraph(df, 0)

    print(train_data)
    print(train_labels)

    nid = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    content = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    sentiment = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    d = {'nid': nid, 'content': content, 'sentiment': sentiment}
    df = pd.DataFrame(data=d)
    train_data, train_labels = clean_train_set_paragraph(df, 0)

    print(train_data)
    print(train_labels)


def test_clean_train_set_paragraph_real_dataset():
    df_data_train = pd.read_csv("../data/SentiNews_paragraph-level.txt", sep="\t")
    nr_articles = sorted(list(set(df_data_train['nid'])))
    len_df = len(nr_articles)
    print(len_df)
    for i in range(10):
        train_data, train_labels = clean_train_set_paragraph(df_data_train, len_df, i)


if __name__ == "__main__":
    test_clean_train_set_paragraph_real_dataset()