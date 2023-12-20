#from modeling_bert import BertLMPredictionHead, BertForPreTraining, BertModel
import sys
sys.path.append("/Users/lukaandrensek/Documents/first_task_on_ijs/sentiment_enrichment/xlmr/transformers-main/src/transformers/models/xlm_roberta")
sys.path.append("/home/lukaa/first_task_on_ijs/sentiment_enrichment/xlmr/transformers-main/src/transformers/models/xlm_roberta")
from modeling_xlm_roberta import XLMRobertaLMHead

from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import XLMRobertaModel 
from transformers import XLMRobertaForMaskedLM
from transformers import BertForPreTraining
"""
class XLMRobertaSentimentPretrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = XLMRobertaLMHead(config) #kaj tu?
        self.sentiment_classification = nn.Linear(config.hidden_size, 3)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        sentiment_classification_score = self.sentiment_classification(pooled_output)
        return prediction_scores, sentiment_classification_score
"""

class XLMRobertatForSentimentPretraining(XLMRobertaForMaskedLM):
    """Based on original huggingface transformers code.
    Source: https://huggingface.co/transformers/_modules/transformers/modeling_bert.html#BertForPreTraining"""
    def __init__(self, config):
        super().__init__(config)

        self.roberta = XLMRobertaModel(config)
        self.lm_head = XLMRobertaLMHead(config)
        self.sentiment_claasification = nn.Linear(config.hidden_size, 3)

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            masked_lm_labels=None,
            next_sentence_label=None,
    ):
        r"""
        masked_lm_labels (``torch.LongTensor`` of shape ``(batch_size, sequence_length)``, `optional`, defaults to :obj:`None`):
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``
        next_sentence_label (``torch.LongTensor`` of shape ``(batch_size,)``, `optional`, defaults to :obj:`None`):
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair (see :obj:`input_ids` docstring)
            Indices should be in ``[0, 1]``.
            ``0`` indicates sequence B is a continuation of sequence A,
            ``1`` indicates sequence B is a random sequence.

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (`optional`, returned when ``masked_lm_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Total loss as the sum of the masked language modeling loss and the next sequence prediction (classification) loss.
        prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`)
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        seq_relationship_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, 2)`):
            Prediction scores of the next sequence prediction (classification) head (scores of True/False
            continuation before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when :obj:`config.output_hidden_states=True`):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.


    Examples::

        from transformers import BertTokenizer, BertForPreTraining
        import torch

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForPreTraining.from_pretrained('bert-base-uncased')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)

        prediction_scores, seq_relationship_scores = outputs[:2]

        """

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output, pooled_output = outputs[:2]
        prediction_scores = self.lm_head(sequence_output)
        sentiment_classification_score = self.sentiment_claasification(pooled_output)
        
        #prediction_scores, seq_relationship_score = self.lm_head(sequence_output, pooled_output)

        outputs = (prediction_scores, sentiment_classification_score,) + outputs[
                                                                 2:
                                                                 ]  # add hidden states and attention if they are here

        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            sentiment_loss = loss_fct(sentiment_classification_score.view(-1, 3), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + sentiment_loss
            outputs = (total_loss,) + outputs

        return outputs  # (loss), prediction_scores, sentiment_clasification_score, (hidden_states), (attentions)
    
