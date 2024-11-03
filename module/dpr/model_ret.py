import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer


class Encoder(nn.Module):
    def __init__(self, dense_model_name_or_path):
        super(Encoder, self).__init__()
        self.model_name = dense_model_name_or_path
        self.config = AutoConfig.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name, config=self.config)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        model_name = self.model_name.split("/")[-1].split("-")[0].lower()
        if model_name == "bert":
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        else:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        return outputs.pooler_output
