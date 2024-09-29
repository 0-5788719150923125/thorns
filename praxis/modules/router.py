import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..configuration_praxis import PraxisConfig


class PraxisMixtureOfDepths(nn.Linear):
    """
    This uses expert-choice routing, which was greatly preferred by the
    original authors of this research: https://arxiv.org/abs/2404.02258
    """

    def __init__(
        self,
        config: PraxisConfig = None,
        *args,
        **kwargs,
    ):
        super().__init__(in_features=config.n_dim, out_features=1)
        self.capacity = config.capacity

    def forward(
        self,
        inputs: Tensor,
        expert: nn.Module,
        attention_mask: Tensor,
        *args,
        **kwargs,
    ):

        b, s, d = inputs.shape

        # scalar weights for each token
        router_logits = F.linear(
            inputs, self.weight
        )  # (x) batch,seq_len,dim -> r batch,seq_len,1

        k = int(s * self.capacity)

        if self.training:
            #  𝑟𝑙> 𝑃𝛽 (R) - equation 1
            token_weights, token_indices = torch.topk(
                # page 7: the [aux] loss centers the sigmoid of the router’s outputs around 0.5
                torch.sigmoid(router_logits),
                k,
                dim=1,
                sorted=False,
            )
        else:
            # top-k breaks causality; the sigmoid here allows us to sample autoregressively during inference
            token_mask = torch.sigmoid(router_logits) > 0.5
            token_indices = torch.nonzero(token_mask, as_tuple=True)[1].view(b, -1)

            if token_indices.numel() == 0:
                # if no tokens were selected, just use the most recent k tokens
                select_tokens = min(k, s)
                token_indices = (
                    torch.arange(s - select_tokens, s, device=inputs.device)
                    .view(1, -1)
                    .expand(b, -1)
                )
                token_weights = torch.ones(b, select_tokens, 1, device=inputs.device)
            else:
                token_weights = (
                    router_logits.squeeze(-1).gather(b, token_indices).unsqueeze(-1)
                )

            token_indices = token_indices.unsqueeze(-1)

        # required to maintain the casual nature of an autoregressive model
        sorted_index_values, sorted_index_indices = torch.sort(token_indices, dim=1)

        # select idx for copying for original tensor
        indices_expanded = sorted_index_values.expand(-1, -1, d)

        # filtered topk tokens with a capacity of C
        filtered_inputs = torch.gather(
            input=inputs, dim=1, index=indices_expanded
        )  # -> batch, capacity, dim

        # selecting router weight by idx
        router_weights = torch.gather(token_weights, dim=1, index=sorted_index_indices)

        # slice the attention mask based on the selected token indices
        filtered_attention_mask = torch.gather(
            input=attention_mask,
            dim=1,
            index=sorted_index_values.squeeze(-1),
        )

        # pass the selected tokens through the transformer block
        expert_outputs = expert(
            filtered_inputs,
            attention_mask=filtered_attention_mask,
            router_weights=router_weights,
        )
        # integrate the selected and residual tokens
        outputs = torch.scatter(
            input=inputs,
            dim=1,
            index=indices_expanded,
            src=expert_outputs["hidden_states"],
        )

        # compute aux loss, in order to maintain causality
        aux_loss = self.aux_loss(router_logits, sorted_index_values)

        return dict(hidden_states=outputs, aux_loss=aux_loss)

    def aux_loss(self, router_logits: torch.Tensor, selected_tokens: torch.Tensor):
        # section 3.5: sampling
        router_targets = torch.zeros_like(router_logits)
        router_targets.scatter_(1, selected_tokens, 1.0)
        return F.binary_cross_entropy_with_logits(
            router_logits.view(-1), router_targets.view(-1)
        )
