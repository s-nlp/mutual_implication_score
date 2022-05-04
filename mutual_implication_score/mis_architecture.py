import torch
from torch import nn
from transformers import RobertaPreTrainedModel, RobertaModel, PreTrainedModel


class RobertaClassificationHeadPruned(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj_adjusted = nn.Linear(config.hidden_size*2, 1)
        
    def forward_direction(self, features):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return x

    def forward(self, features_forward, features_backward, **kwargs):
        
        x_forward = self.forward_direction(features_forward)
        x_backward = self.forward_direction(features_backward)
        
        concatenated_hs = torch.cat([x_forward, x_backward], -1)
                
        x = self.out_proj_adjusted(concatenated_hs)
        return x


class TwoFoldRoberta(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = RobertaClassificationHeadPruned(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        batch_forward, batch_backward,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        outputs_forward = self.roberta(
            **batch_forward
        )[0]
        
        outputs_backward = self.roberta(
            **batch_backward
        )[0]
                
        logits = self.classifier(outputs_forward, outputs_backward)
                
        return logits
