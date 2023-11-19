import torch
from transformers import AdamW, get_linear_schedule_with_warmup

from tqdm import tqdm, trange
import os
import sys
import numpy as np
from metrics import get_metrics


def xlmr_train(model, device, train_dataloader, eval_dataloader, output_dir, num_epochs, warmup_proportion, weight_decay,
               learning_rate, adam_epsilon, save_best=False):
    """Training loop for bert fine-tuning. Save best works with F1 only currently."""

    t_total = len(train_dataloader) * num_epochs
    warmup_steps = len(train_dataloader) * warmup_proportion
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=t_total)
    train_iterator = trange(int(num_epochs), desc="Epoch")
    model.to(device)
    tr_loss_track = []
    eval_metric_track = []
    output_filename = os.path.join(output_dir, 'pytorch_model.bin')
    f1 = float('-inf')

    for _ in train_iterator:
        model.train()
        model.zero_grad()
        tr_loss = 0
        nr_batches = 0
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            tr_loss = 0
            input_ids, input_mask, labels = batch
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=input_mask, labels=labels)
            loss = outputs[0]
            loss.backward()
            optimizer.step()
            scheduler.step()
            tr_loss += loss.item()
            nr_batches += 1
            model.zero_grad()

        print("Evaluating the model on the evaluation split...")
        metrics = bert_evaluate(model, eval_dataloader, device)
        eval_metric_track.append(metrics)
        if save_best:
            if f1 < metrics['f1']:
                model.save_pretrained(output_dir)
                torch.save(model.state_dict(), output_filename)
                print("The new value of f1 score of " + str(metrics['f1']) + " is higher then the old value of " +
                      str(f1) + ".")
                print("Saving the new model...")
                f1 = metrics['f1']
            else:
                print("The new value of f1 score of " + str(metrics['f1']) + " is not higher then the old value of " +
                      str(f1) + ".")

        tr_loss = tr_loss / nr_batches
        tr_loss_track.append(tr_loss)

    if not save_best:
        model.save_pretrained(output_dir)
        torch.save(model.state_dict(), output_filename)

    return tr_loss_track, eval_metric_track


def xlmr_evaluate(model, eval_dataloader, device):
    """Evaluation of trained checkpoint."""
    model.to(device)
    model.eval()
    predictions = []
    true_labels = []
    data_iterator = tqdm(eval_dataloader, desc="Iteration")
    for step, batch in enumerate(data_iterator):
        input_ids, input_mask, labels = batch
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)

        with torch.no_grad():
            outputs = model(input_ids, token_type_ids=None, attention_mask=input_mask)

        #loss is only output when labels are provided as input to the model
        logits = outputs[0]
        print(type(logits))
        logits = logits.to('cpu').numpy()
        label_ids = labels.to('cpu').numpy()

        for label, logit in zip(label_ids, logits):
            true_labels.append(label)
            predictions.append(np.argmax(logit))


    metrics = get_metrics(true_labels, predictions)
    return metrics


def xlmr_train_lm(model, device, train_dataloader, output_dir, num_epochs, warmup_proportion, weight_decay,
               learning_rate, adam_epsilon, save_best=False, eval_dataloader=None):
    """Training loop for bert fine-tuning. Save best works with F1 only currently."""

    t_total = len(train_dataloader) * num_epochs
    warmup_steps = len(train_dataloader) * warmup_proportion
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=t_total)
    train_iterator = trange(int(num_epochs), desc="Epoch")
    model.to(device)
    tr_loss_track = []
    num_iterations = 0
    early_stopping = 0

    output_filename = os.path.join(output_dir, 'pytorch_model.bin')
    perplexity_history = float('inf')

    for _ in train_iterator:
        model.train()
        model.zero_grad()
        tr_loss = 0
        nr_batches = 0
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            input_ids, input_mask, labels, mlm_labels = batch
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            labels = labels.to(device)
            mlm_labels = mlm_labels.to(device)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=input_mask, masked_lm_labels=mlm_labels,
                            next_sentence_label=labels)
            loss = outputs[0]
            loss.backward()
            optimizer.step()
            scheduler.step()
            #tr_loss += loss.detach().item()
            nr_batches += 1
            model.zero_grad()
            del loss
            del input_ids
            del input_mask
            del labels
            del mlm_labels

        if save_best:
            if eval_dataloader == None:
                print("Please provide evaluation data.")
                sys.exit()
            perplexity = xlmr_evaluate_lm(model, eval_dataloader, device)
            print(type(perplexity))
            if perplexity_history > perplexity:
                model.save_pretrained(output_dir)
                torch.save(model.state_dict(), output_filename)
                print("The new value of perplexity of " + str(perplexity) + " is lower then the old value of " +
                      str(perplexity_history) + ".")
                print("Saving the new model...")
                perplexity_history = perplexity
            else:
                print("The new value of perplexity of " + str(perplexity) + " is not lower then the old value of " +
                      str(perplexity_history) + ".")

            if (perplexity_history < perplexity) and early_stopping == 1:
                break
            elif (perplexity_history < perplexity) and early_stopping == 0:
                early_stopping = 1
            elif (perplexity_history > perplexity) and early_stopping == 1:
                early_stopping = 0

        tr_loss = tr_loss / nr_batches
        tr_loss_track.append(tr_loss)
        num_iterations += 1


    if not save_best:
        model.save_pretrained(output_dir)
        # tokenizer.save_pretrained(output_dir)
        torch.save(model.state_dict(), output_filename)

    return tr_loss_track, num_iterations


def xlmr_evaluate_lm(model, eval_dataloader, device):
    """Evaluation of trained checkpoint."""
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    data_iterator = tqdm(eval_dataloader, desc="Iteration")
    for step, batch in enumerate(data_iterator):
        input_ids, input_mask, labels, mlm_labels = batch
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        labels = labels.to(device)
        mlm_labels = mlm_labels.to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=input_mask, masked_lm_labels=mlm_labels,
                            next_sentence_label=labels)
            loss = outputs[0]
            eval_loss += loss.detach().mean().item()
            del loss
        nb_eval_steps += 1
        del input_ids
        del input_mask
        del labels
        del mlm_labels

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))
    perplexity = perplexity.item()

    return perplexity


def xlmr_extract_s_embedding(model, dataloader, device):
    """For each datapoint extracts embeddings for <s> token from the last layer and returns a list of <s> embeddings
    NOTE: input model should have the option 'output_hidden_states' set to True when initialized
    """
    model.to(device)
    model.eval()
    s_embedding = []
    for step, batch in enumerate(dataloader):
        input_ids, input_mask = batch
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)

        with torch.no_grad():
            outputs = model(input_ids, token_type_ids=None, attention_mask=input_mask)

        all_hidden_states = outputs[-1]
        last_layer = all_hidden_states[-1] #### [0], ker last_layer je (tensor(...))
        last_layer = last_layer.to('cpu').numpy()
        print("Shape:")
        print(last_layer.shape)
        s_embedding.append(last_layer[0])
        print(s_embedding.shape)

    return s_embedding

