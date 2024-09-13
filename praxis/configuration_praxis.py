from transformers import PretrainedConfig


class PraxisConfig(PretrainedConfig):
    model_type = "praxis"

    def __init__(
        self,
        n_embd=768,
        n_layer=12,
        n_head=12,
        activation_function="mish",
        attn_pdrop=0,
        resid_pdrop=0,
        embd_pdrop=0,
        rms_norm_epsilon=1e-5,
        initializer_range=0.02,
        n_experts=3,
        k_best=2,
        target_temperature=0.9,
        annealing_steps=10_000,
        vocab_size=32000,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        unk_token_id=4,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            unk_token_id=unk_token_id,
            **kwargs,
        )

        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.activation_function = activation_function
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.rms_norm_epsilon = rms_norm_epsilon
        self.initializer_range = initializer_range
        self.n_experts = n_experts
        self.k_best = k_best
        self.target_temperature = target_temperature
        self.annealing_steps = annealing_steps
        self.use_cache = False
        self.causal = False
