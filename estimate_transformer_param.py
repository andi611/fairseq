import torch
from torch import nn


# https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9
def count_parameters(model: torch.nn.Module) -> int:
    """ Returns the number of learnable parameters for a PyTorch model """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

layer = 12
d_model = 360
d_ff = 768
n_heads = 8
final_dim = 256
num_classes = 500

from fairseq.models.wav2vec.wav2vec2 import TransformerSentenceEncoderLayer
def build_encoder_layer():
    return TransformerSentenceEncoderLayer(
        embedding_dim=d_model,
        ffn_embedding_dim=d_ff,
        num_attention_heads=n_heads,
    )
encoder_layers = nn.ModuleList(
            [build_encoder_layer() for _ in range(layer)]
        )

from fairseq.models.wav2vec.wav2vec2 import ConvFeatureExtractionModel
feature_extractor = ConvFeatureExtractionModel(
            conv_layers=eval("[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2"),
            dropout=0.0,
            mode='default',
            conv_bias=False,
        )

target_glu = nn.Sequential(nn.Linear(final_dim, final_dim * 2), nn.GLU())

predictor_count = num_classes * final_dim
final_proj_count = count_parameters(nn.Linear(d_model, final_dim))
target_glu_count = count_parameters(target_glu)
enocder_count = count_parameters(encoder_layers)
cnn_count = count_parameters(feature_extractor)
mask_emb_count = d_model

print("Predictor:", predictor_count)
print("Projector:", final_proj_count)
print("Encoder:", enocder_count)
print("CNN:", cnn_count)
print("Total:", predictor_count+final_proj_count+target_glu_count+enocder_count+cnn_count+mask_emb_count)