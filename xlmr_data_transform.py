import torch
import random
from keras_preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, RandomSampler, DataLoader
import math


def random_word(tokens, tokenizer):
    """
    Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
    Source: https://github.com/huggingface/transformers/pull/124/files
    :param tokens: list of str, tokenized sentence.
    :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
    :return: (list of str, list of int), masked tokens and related labels for LM prediction
    """
    output_label = []

    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = "<mask>"

            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.choice(list(tokenizer.vocab.items()))[0]

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later)
            try:
                output_label.append(tokenizer.vocab[token])
            except KeyError:
                # For unknown words (should not occur with BPE vocab)
                output_label.append(tokenizer.vocab["<unk>"])
                #logger.warning("Cannot find token '{}' in vocab. Using [UNK] insetad".format(token))
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-100)

    return tokens, output_label


def prepare_data_for_language_modelling(data, labels, tokenizer, max_len, batch_size):
    """REFACTOR: MAKE TWO FUNCTIONS - ONE PREPARES DATA, THE OTHER CREATES A DATALOADER"""
    tokenized_sentences = [tokenizer.tokenize(sentence) for sentence in data]
    truncated_sentences = [sentence[:(max_len - 2)] for sentence in tokenized_sentences]
    masked_sentences = []
    mlm_labels = []
    i = 0 
    l = len(truncated_sentences)
    for sentence in truncated_sentences:
        print(i, l)
        i += 1
        m_sent, mlm_l = random_word(sentence, tokenizer)
        masked_sentences.append(m_sent)
        mlm_labels.append(mlm_l)
    masked_sentences_with_special_tokens = [["<s>"] + sentence + ["</s>"] for sentence in masked_sentences]
    # adding ids for <s> and </s> special tokens, to be ignored by the loss function
    final_mlm_labels = []
    for mlm_label in mlm_labels:
        final_mlm_labels.append([-100] + mlm_label + [-100])

    #tokenized_sentences = [tokenizer.tokenize(sentence) for sentence in sentences]
    print("Example of tokenized sentence:")
    print(masked_sentences_with_special_tokens[0])

    input_ids = [tokenizer.convert_tokens_to_ids(sentence) for sentence in masked_sentences_with_special_tokens]
    print("Printing encoded sentences:")
    print(input_ids[0])
    print("Printing mlm_labels:")
    print(final_mlm_labels[0])
    # dtype must be long because BERT apparently expects it
    #input_ids = pad_sequences(input_ids, dtype='long', maxlen=max_len, padding="post", truncating="post")
    #final_mlm_labels = pad_sequences(final_mlm_labels, dtype='long', maxlen=max_len, padding="post", truncating="post",
    #                                 value=-100)
    #zero-padding
    for input, label in zip(input_ids, final_mlm_labels):
        while len(input) < max_len:
            input.append(1)
            label.append(-100)

    # attention masks
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i != 1) for i in seq]
        attention_masks.append(seq_mask)
    #print(len(input_ids[0]))
    #print(len(final_mlm_labels[0]))
    input_ids = torch.tensor(input_ids)
    labels = torch.tensor(labels)
    final_mlm_labels = torch.tensor(final_mlm_labels)
    attention_masks = torch.tensor(attention_masks)

    transformed_data = TensorDataset(input_ids, attention_masks, labels, final_mlm_labels)
    sampler = RandomSampler(transformed_data)
    dataloader = DataLoader(transformed_data, sampler=sampler, batch_size=batch_size)

    return dataloader


def cut_at_length(data, labels, tokenizer, max_len, batch_size):
    """Shortens each example to the length, specified by max_len parameter.
    Returns a dataloader with labels
    REFACTOR: MAKE TWO FUNCTIONS - ONE PREPARES DATA, THE OTHER CREATES A DATALOADER"""
    #sentences = ["[CLS] " + sentence + " [SEP]" for sentence in data]

    tokenized_sentences = [tokenizer.tokenize(sentence) for sentence in data]
    truncated_sentences = [sentence[:(max_len - 2)] for sentence in tokenized_sentences]
    truncated_sentences = [["<s>"] + sentence + ["</s>"] for sentence in truncated_sentences]
    print("Example of tokenized sentence:")
    print(truncated_sentences[0])

    input_ids = [tokenizer.convert_tokens_to_ids(sentence) for sentence in truncated_sentences]
    print("Printing encoded sentences:")
    print(input_ids[0])
    # dtype must be long because BERT apparently expects it
    input_ids = pad_sequences(input_ids, dtype='long', maxlen=max_len, padding="post", truncating="post", value=1.0)

    # attention masks
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i != 1) for i in seq]
        attention_masks.append(seq_mask)

    input_ids = torch.tensor(input_ids)
    labels = torch.tensor(labels)
    attention_masks = torch.tensor(attention_masks)

    transformed_data = TensorDataset(input_ids, attention_masks, labels)
    sampler = RandomSampler(transformed_data)
    dataloader = DataLoader(transformed_data, sampler=sampler, batch_size=batch_size)

    return dataloader


def encode_labels(labels, labels_set):
    """Maps each label to a unique index.
    :param labels: (list of strings) labels of every instance in the dataset
    :param labels_set: (list of strings) set of labels that appear in the dataset
    :return (list of int) encoded labels
    """
    encoded_labels = []
    for label in labels:
        encoded_labels.append(labels_set.index(label))
    return encoded_labels


def cut_at_front_and_back(data, labels, tokenizer, max_len, batch_size):
    """REFACTOR: MAKE TWO FUNCTIONS - ONE PREPARES DATA, THE OTHER CREATES A DATALOADER"""
    sentences = ["<s> " + sentence for sentence in data]

    tokenized_sentences = [tokenizer.tokenize(sentence) for sentence in sentences]

    cut_tokenized_sentences = []
    for tokenized_sentence in tokenized_sentences:
        if len(tokenized_sentence) < max_len:
            cut_tokenized_sentences.append(tokenized_sentence + ["</s>"])
        elif len(tokenized_sentence) > max_len:
            tokenized_sentence = tokenized_sentence[:math.floor(max_len / 2)] + \
                           tokenized_sentence[-(math.ceil(max_len / 2) - 1):] + ["</s>"]
            cut_tokenized_sentences.append(tokenized_sentence)
        else:
            tokenized_sentence = tokenized_sentence[:-1] + ["</s>"]
            cut_tokenized_sentences.append(tokenized_sentence)

    print("Example of tokenized sentence:")
    print(cut_tokenized_sentences[0])

    input_ids = [tokenizer.convert_tokens_to_ids(sentence) for sentence in cut_tokenized_sentences]
    print("Printing encoded sentences:")
    print(input_ids[0])

    # dtype must be long because BERT apparently expects it
    input_ids = pad_sequences(input_ids, dtype='long', maxlen=max_len, padding="post", truncating="post", value=1.0)

    #print("Printing cut sequence:")
    #print(cut_input_ids[0])
    # attention masks
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i != 1) for i in seq]
        attention_masks.append(seq_mask)

    input_ids = torch.tensor(input_ids)
    labels = torch.tensor(labels)
    attention_masks = torch.tensor(attention_masks)

    transformed_data = TensorDataset(input_ids, attention_masks, labels)
    sampler = RandomSampler(transformed_data)
    dataloader = DataLoader(transformed_data, sampler=sampler, batch_size=batch_size)

    return dataloader
