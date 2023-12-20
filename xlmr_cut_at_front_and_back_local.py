from xlmr_ml_sentiment_classifier import xlmr_train, xlmr_evaluate
from xlmr_data_transform import encode_labels, cut_at_front_and_back
from read_data import read_croatian_data

from transformers import XLMRobertaConfig
from transformers import XLMRobertaForSequenceClassification
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import torch
import pandas as pd
import numpy as np

import random
import argparse
import os
from math import floor

def crossvalidation_front_back():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_data_path",
                        required=True,
                        type=str)
    parser.add_argument("--output_dir",
                        required=True,
                        type=str)
    parser.add_argument("--cro_test_data_path",
                        type=str)

    parser.add_argument("--do_lower_case",
                        action='store_true')
    #parser.add_argument("--split_num",
    #                    default=2,
    #                    type=int)
    #parser.add_argument("--config_file",
    #                    type=str)
    #parser.add_argument("--model_file",
    #                    type=str)
    parser.add_argument("--eval_split",
                        default=0.2,
                        type=float)
    parser.add_argument("--test_split",
                        default=0.1,
                        type=float)
    parser.add_argument("--max_len",
                        default=512,
                        type=int)
    parser.add_argument("--batch_size",
                        default=16,
                        type=int)
    parser.add_argument("--num_epochs",
                        default=3,
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
        print("--------------------------")
        print("Training model :", i)
        
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        output_dir = os.path.join(args.output_dir, f"sentiment_language_model_{i}")
        log_path = os.path.join(output_dir, f"log")
        
        
        print("Reading data...")
        df_data = pd.read_csv(args.train_data_path, sep="\t")
        
        print("Number of instances:", len(df_data))
        
        
        data = df_data['content'].tolist()
        label_set = sorted(list(set(df_data['sentiment'].values)))
        labels = encode_labels(df_data['sentiment'].tolist(), label_set)
        

        if args.cro_test_data_path is not None:
            print("Preparing the croatian test data...")
            cro_test_data, cro_test_labels = read_croatian_data(args.cro_test_data_path)
            cro_test_labels = encode_labels(cro_test_labels, label_set)


        print("Training model on the split number " + str(i) + "...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base', do_lower_case=args.do_lower_case) ####ne škodi, če pustimo do_lower_case
        
        ###
        config_file = f"/Users/lukaandrensek/Documents/first_task_on_ijs/sentiment_enrichment/models/sentiment_language_model_{i}/config.json"
        model_file = f"/Users/lukaandrensek/Documents/first_task_on_ijs/sentiment_enrichment/models/sentiment_language_model_{i}/pytorch_model.bin"
        ###
        
        if config_file is not None and model_file is not None:
            config = XLMRobertaConfig.from_pretrained(config_file,
                                                num_labels=len(label_set))
            model = XLMRobertaForSequenceClassification.from_pretrained(model_file,
                                                                config=config)
            print("We loaded in the weights of the model", i)
            print("config file:", config_file)
            print("model file:", model_file)
        else:
            model = XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-base',
                                                                num_labels=len(label_set))
            print("We loaded in fresh weights.")

        test_data = data[(floor(len(data) * i * 0.1)):(floor(len(data) * (i + 1) * 0.1))]
        test_labels = labels[floor((len(labels) * i * 0.1)):floor((len(labels) * (i + 1) * 0.1))]
        train_data = data[:floor((len(data) * i * 0.1))] + data[floor((len(data) * (i + 1) * 0.1)):]
        train_labels = labels[:floor((len(labels) * i * 0.1))] + labels[floor((len(labels) * (i + 1) * 0.1)):]
        train_data, eval_data, train_labels, eval_labels = train_test_split(train_data, train_labels,
                                                                    test_size=args.eval_split, random_state=42)
        print("Train label:")
        print(train_labels[0])
        print("Train data:")
        print(train_data[0])
        train_dataloader = cut_at_front_and_back(train_data, train_labels, tokenizer, args.max_len, args.batch_size)
        eval_dataloader = cut_at_front_and_back(eval_data, eval_labels, tokenizer, args.max_len, args.batch_size)
        test_dataloader = cut_at_front_and_back(test_data, test_labels, tokenizer, args.max_len, args.batch_size)
        if args.cro_test_data_path is not None:
            cro_test_dataloader = cut_at_front_and_back(cro_test_data, cro_test_labels, tokenizer, args.max_len, args.batch_size)
        _, __ = xlmr_train(model, device, train_dataloader, eval_dataloader, output_dir, args.num_epochs,
                        args.warmup_proportion, args.weight_decay, args.learning_rate, args.adam_epsilon,
                        save_best=True)

        print(f"Testing the trained model {i} on the current test split...")
        metrics = xlmr_evaluate(model, test_dataloader, device)
        with open(log_path, 'a') as f:
            f.write("Results for split nr. " + str(i) + " on current slo test:\n")
            f.write("Acc: " + str(metrics['accuracy']) + "\n")
            f.write("Recall: " + str(metrics['recall']) + "\n")
            f.write("Precision: " + str(metrics['precision']) + "\n")
            f.write("F1: " + str(metrics['f1']) + "\n")
            f.write("\n")

        if args.cro_test_data_path is not None:
            print(f"Testing the trained model {i} on the croatian test set...")
            cro_metrics = xlmr_evaluate(model, cro_test_dataloader, device)
            with open(log_path, 'a') as f:
                f.write("Results for split nr. " + str(i) + " on cro test set:\n")
                f.write("Acc: " + str(cro_metrics['accuracy']) + "\n")
                f.write("Recall: " + str(cro_metrics['recall']) + "\n")
                f.write("Precision: " + str(cro_metrics['precision']) + "\n")
                f.write("F1: " + str(cro_metrics['f1']) + "\n")
                f.write("\n")
        print("Done.")


if __name__ == "__main__":
    crossvalidation_front_back()
#/Users/lukaandrensek/Documents/first_task_on_ijs/sentiment_enrichment/finished_models
#/Users/lukaandrensek/Documents/first_task_on_ijs/datasets/SentiNews_document-level.txt
#/Users/lukaandrensek/Documents/first_task_on_ijs/datasets/croatian_sentiment_news_document.tsv
#python3 xlmr_cut_at_front_and_back_local.py --train_data_path /Users/lukaandrensek/Documents/first_task_on_ijs/datasets/SentiNews_document-level.txt --output_dir /Users/lukaandrensek/Documents/first_task_on_ijs/sentiment_enrichment/finished_models --cro_test_data_path /Users/lukaandrensek/Documents/first_task_on_ijs/datasets/crosslingual_test_data/croatian_test_bucar
#python3 xlmr_cut_at_front_and_back_copy.py --train_data_path /home/lukaa/first_task_on_ijs/datasets/SentiNews_document-level.txt --output_dir /home/lukaa/first_task_on_ijs/sentiment_enrichment/finished_models --cro_test_data_path /home/lukaa/first_task_on_ijs/datasets/crosslingual_test_data/croatian_test_bucar