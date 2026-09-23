from fuxictr.pytorch.layers import FactorizationMachine, MLP_Block
from fuxictr.pytorch.models import BaseModel

from .numerical_embeddings import MixedFeatureEmbedding


class NumEmbDeepFM(BaseModel):
    """DeepFM backbone with selectable numerical embeddings."""

    def __init__(
        self,
        feature_map,
        model_id="NumEmbDeepFM",
        gpu=-1,
        learning_rate=1e-3,
        embedding_dim=16,
        numerical_embedding="field",
        numerical_embedding_params=None,
        hidden_units=(256, 128, 64),
        hidden_activations="ReLU",
        net_dropout=0,
        batch_norm=False,
        lr_use_shared_numeric_embedding=False,
        embedding_regularizer=None,
        net_regularizer=None,
        **kwargs,
    ):
        super().__init__(
            feature_map,
            model_id=model_id,
            gpu=gpu,
            embedding_regularizer=embedding_regularizer,
            net_regularizer=net_regularizer,
            **kwargs,
        )
        self.embedding_layer = MixedFeatureEmbedding(
            feature_map,
            embedding_dim,
            numerical_embedding=numerical_embedding,
            numerical_embedding_params=numerical_embedding_params,
        )
        self.fm = FactorizationMachine(
            feature_map,
            numeric_embedding_dim=(
                embedding_dim if lr_use_shared_numeric_embedding else None
            ),
        )
        self.mlp = MLP_Block(
            input_dim=feature_map.sum_emb_out_dim(),
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activations=hidden_activations,
            output_activation=None,
            dropout_rates=net_dropout,
            batch_norm=batch_norm,
        )
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        x = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(x)
        y_pred = self.fm(x, feature_emb)
        y_pred = y_pred + self.mlp(feature_emb.flatten(start_dim=1))
        return {"y_pred": self.output_activation(y_pred)}
