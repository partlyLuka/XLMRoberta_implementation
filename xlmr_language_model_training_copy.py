from xlmr_ml_sentiment_classifier import xlmr_train_lm
from xlmr_data_transform import encode_labels, prepare_data_for_language_modelling
from XLMRSentimentPretraining import XLMRobertatForSentimentPretraining
from read_data import clean_train_set_paragraph

from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import torch
import pandas as pd
import numpy as np

import random
import argparse
import os

def sentiment_enrichment():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_data_path",
                        required=True,
                        type=str)
    parser.add_argument("--output_dir",
                        required=True,
                        type=str)
    parser.add_argument("--do_lower_case",
                        action='store_true')
    parser.add_argument("--eval_split",
                        default=0.1,
                        type=float)
    #parser.add_argument("--split_num",
    #                    default=2,
    #                    type=int)
    parser.add_argument("--max_len",
                        default=512,
                        type=int)
    parser.add_argument("--batch_size",
                        default=16,
                        type=int)
    parser.add_argument("--num_epochs",
                        default=5,
                        type=int)
    parser.add_argument("--learning_rate",
                        default=2e-5,
                        type=float)
    parser.add_argument("--weight_decay",
                        default=0.01,
                        type=float)
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float)
    parser.add_argument("--adam_epsilon",
                        default=1e-8,
                        type=float)

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    print("Setting the random seed...")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    for i in range(10):
        print(" ")
        print("-------------------------------")
        print("Training model :", i)
        output_dir = os.path.join(args.output_dir, f"sentiment_language_model_{i}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print("Reading data...")
        df_data_train = pd.read_csv(args.train_data_path, sep="\t")
        
        print("Number of instances:", len(df_data_train))
        
        nr_articles = sorted(list(set(df_data_train['nid'])))
        nr_articles = len(nr_articles)
        train_data, train_labels = clean_train_set_paragraph(df_data_train, nr_articles, i)
        label_set = sorted(list(set(df_data_train['sentiment'].values)))
        train_labels = encode_labels(train_labels, label_set)
        num_labels = len(label_set)

        train_data, eval_data, train_labels, eval_labels = train_test_split(train_data, train_labels,
                                                                            test_size=args.eval_split, random_state=42)

        print("XLMR pretraining as a language model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Device:", device)
        tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base', do_lower_case=args.do_lower_case)
        model = XLMRobertatForSentimentPretraining.from_pretrained('xlm-roberta-base', num_labels=num_labels)
        print("Train label:")
        print(train_labels[0])
        print("Train data:")
        print(train_data[0])
        
        train_dataloader = prepare_data_for_language_modelling(train_data, train_labels, tokenizer, args.max_len,
                                                            args.batch_size)
        
        eval_dataloader = prepare_data_for_language_modelling(eval_data, eval_labels, tokenizer, args.max_len,
                                                            args.batch_size)
        tr_loss_track, num_iterations = xlmr_train_lm(model, device, train_dataloader, output_dir, args.num_epochs,
                        args.warmup_proportion, args.weight_decay, args.learning_rate, args.adam_epsilon,
                        save_best=True, eval_dataloader=eval_dataloader)
        print("Done.")


if __name__ == "__main__":
    sentiment_enrichment()
#/Users/lukaandrensek/Documents/first_task_on_ijs/sentiment_enrichment/models
#/Users/lukaandrensek/Documents/first_task_on_ijs/datasets/SentiNews_paragraph-level.txt
#python3 xlmr_language_model_training_copy.py --train_data_path /Users/lukaandrensek/Documents/first_task_on_ijs/datasets/SentiNews_paragraph-level.txt --output_dir /Users/lukaandrensek/Documents/first_task_on_ijs/sentiment_enrichment/models
#python3 xlmr_language_model_training_copy.py --train_data_path /home/lukaa/first_task_on_ijs/datasets/SentiNews_paragraph-level.txt --output_dir /home/lukaa/first_task_on_ijs/sentiment_enrichment/models