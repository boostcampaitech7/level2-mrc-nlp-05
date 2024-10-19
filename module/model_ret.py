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
        if token_type_ids is not None:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids if "bert" in model_name else None,
            )
        else:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        if hasattr(outputs, "pooler_output"):
            return outputs.pooler_output
        else:
            return outputs.last_hidden_state[:, 0, :]
