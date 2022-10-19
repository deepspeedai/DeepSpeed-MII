'''
Copyright 2022 The Microsoft DeepSpeed Team
'''
from deepspeed.module_inject.base_policy import InjectBasePolicy


class HFGPT2LayerPolicy(InjectBasePolicy):
    _orig_layer_class = None

    def __init__(self, client_module, inference=True):
        # HuggingFace GPT2 uses convolutional layer instead of linear layer
        super().__init__(inference, linear_layer=False)
        self.client_module = client_module
        try:
            import transformers
            HFGPT2LayerPolicy._orig_layer_class = transformers.models.gpt2.modeling_gpt2.GPT2Block
        except:
            HFGPT2LayerPolicy._orig_layer_class = None

    def get_hidden_heads(self):
        return self.client_module.attn.embed_dim, \
                self.client_module.attn.num_heads

    def attention(self):
        return self.linear_layer, \
                self.client_module.attn.c_attn.weight, \
                self.client_module.attn.c_attn.bias, \
                self.client_module.attn.c_proj.weight, \
                self.client_module.attn.c_proj.bias, \
                self.scale_attention, \
                self.is_megatron_v2

    def mlp(self):
        return self.linear_layer, \
            self.client_module.mlp.c_fc.weight, \
            self.client_module.mlp.c_fc.bias, \
            self.client_module.mlp.c_proj.weight, \
            self.client_module.mlp.c_proj.bias

    def layerNorm(self):
        return self.client_module.ln_2.weight, \
               self.client_module.ln_2.bias, \
               self.client_module.ln_1.weight, \
               self.client_module.ln_1.bias
