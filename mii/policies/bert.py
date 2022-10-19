'''
Copyright 2022 The Microsoft DeepSpeed Team
'''
import torch
from torch.nn.parameter import Parameter
from deepspeed.module_inject.base_policy import InjectBasePolicy


class HFBertLayerPolicy(InjectBasePolicy):
    _orig_layer_class = None

    def __init__(self, client_module, inference=False, preln=False):
        super().__init__(inference)
        self.client_module = client_module
        self.preln = preln
        if HFBertLayerPolicy._orig_layer_class is None:
            try:
                import transformers
                HFBertLayerPolicy._orig_layer_class = [
                    transformers.models.bert.modeling_bert.BertLayer,
                    transformers.models.roberta.modeling_roberta.RobertaLayer
                ]
            except:
                HFBertLayerPolicy._orig_layer_class = None

    def get_hidden_heads(self):
        return self.client_module.attention.self.query.weight.shape[1], \
                self.client_module.attention.self.num_attention_heads

    def attention(self):
        qw = self.client_module.attention.self.query.weight
        qb = self.client_module.attention.self.query.bias
        kw = self.client_module.attention.self.key.weight
        kb = self.client_module.attention.self.key.bias
        vw = self.client_module.attention.self.value.weight
        vb = self.client_module.attention.self.value.bias

        qkvw = Parameter(torch.cat((qw, kw, vw), dim=0), requires_grad=False)
        qkvb = Parameter(torch.cat((qb, kb, vb), dim=0), requires_grad=False)

        return self.linear_layer, \
               qkvw, \
               qkvb, \
               self.client_module.attention.output.dense.weight, \
               self.client_module.attention.output.dense.bias, \
               self.scale_attention, \
               self.is_megatron_v2

    def mlp(self):
        if self.preln:
            intermediate_ff = self.client_module.intermediate.dense_act
        else:
            intermediate_ff = self.client_module.intermediate.dense

        return self.linear_layer, intermediate_ff.weight, intermediate_ff.bias, \
            self.client_module.output.dense.weight, \
            self.client_module.output.dense.bias

    def layerNorm(self):
        if self.preln:
            attention_layernorm = self.client_module.PostAttentionLayerNorm
            transformer_layernorm = self.client_module.PreAttentionLayerNorm
        else:
            attention_layernorm = self.client_module.attention.output.LayerNorm
            transformer_layernorm = self.client_module.output.LayerNorm
        return attention_layernorm.weight, \
               attention_layernorm.bias, \
               transformer_layernorm.weight, \
               transformer_layernorm.bias
