from xlmr_data_transform import prepare_data_for_language_modelling, cut_at_length, cut_at_front_and_back
from transformers import AutoTokenizer, XLMRobertaTokenizer
xt = AutoTokenizer.from_pretrained("xlm-roberta-base")
t = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
data = [
    "pozdravljeni ljudje, kje ste ",
    "My kindest names are given above lol",
    "no u lol"
    
]
labels = [0, 0, 1]
load = prepare_data_for_language_modelling(data, labels, tokenizer=t, max_len=100, batch_size=1)

"""
load = cut_at_length(data, labels, tokenizer=xt, max_len=50, batch_size=1)
for b in load:
    print(b)
    break

data = ["1 2 3 4 5 6 7"]
load = cut_at_front_and_back(data, labels=[0], tokenizer=xt, batch_size=1, max_len=16)
for b in load:
    print(b)
"""