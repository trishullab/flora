import math
import re
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Union

import bitsandbytes as bnb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from accelerate import Accelerator
from transformers.pytorch_utils import Conv1D

from ..import_utils import is_bnb_4bit_available, is_bnb_available
from ..utils import (
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    PeftConfig,
    PeftType,
    _freeze_adapter,
    _get_submodules,
    transpose,
)


if is_bnb_available():
    import bitsandbytes as bnb


@dataclass
class FLoraConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`FLoraModel`].

    Args:
        r (`int`): FLora attention dimension.
        target_modules (`Union[List[str],str]`): The names of the modules to apply flora to.
        lora_alpha (`float`): The alpha parameter for Lora scaling.
        lora_dropout (`float`): The dropout probability for Lora layers.
        fan_in_fan_out (`bool`): Set this to True if the layer to replace stores weight like (fan_in, fan_out).
        For example, gpt-2 uses `Conv1D` which stores weights like (fan_in, fan_out) and hence this should be set to `True`.:
        bias (`str`): Bias type for Lora. Can be 'none', 'all' or 'lora_only'
        modules_to_save (`List[str]`):List of modules apart from LoRA layers to be set as trainable
            and saved in the final checkpoint.
    """

    r: int = field(default=1, metadata={"help": "Lora attention dimension"})
    num_heads: int = field(default=1, metadata={"help": "BE number of heads"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    be_scaling: int = field(default=None, metadata={"help": "Lora alpha"})
    be_dropout: float = field(default=None, metadata={"help": "Lora dropout"})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    bias: str = field(default="none", metadata={"help": "Bias type for Lora. Can be 'none', 'all' or 'lora_only'"})
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )
    init_be_weights: bool = field(
        default=True,
        metadata={"help": "Whether to initialize the weights of the Lora layers."},
    )
    fix_alpha: bool = field(
        default=False,
        metadata={"help": "whether fix the alpha adapter."},
    )

    def __post_init__(self):
        self.peft_type = PeftType.FLORA


class FLoraModel(torch.nn.Module):
    """
    Creates Low Rank Adapter (Lora) model from a pretrained transformers model.

    Args:
        model ([`~transformers.PreTrainedModel`]): The model to be adapted.
        config ([`LoraConfig`]): The configuration of the Lora model.

    Returns:
        `torch.nn.Module`: The Lora model.

    **Attributes**:
        - **model** ([`~transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`LoraConfig`]): The configuration of the Lora model.
    """

    def __init__(self, model, config, adapter_name):
        super().__init__()
        self.model = model
        self.forward = self.model.forward
        self.peft_config = config
        self.add_adapter(adapter_name, self.peft_config[adapter_name])

    def add_adapter(self, adapter_name, config=None):
        if config is not None:
            model_config = self.model.config.to_dict() if hasattr(self.model.config, "to_dict") else self.model.config
            config = self._prepare_lora_config(config, model_config)
            self.peft_config[adapter_name] = config
        self._find_and_replace(adapter_name)
        if len(self.peft_config) > 1 and self.peft_config[adapter_name].bias != "none":
            raise ValueError(
                "FLora supports only 1 adapter with bias. When using multiple adapters, set bias to 'none' for all adapters."
            )
        mark_only_be_as_trainable(self.model, self.peft_config[adapter_name].bias)
        if self.peft_config[adapter_name].inference_mode:
            _freeze_adapter(self.model, adapter_name)

    def _find_and_replace(self, adapter_name):
        lora_config = self.peft_config[adapter_name]
        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if (loaded_in_4bit or loaded_in_8bit) and not is_bnb_available():
            raise ImportError(
                "To use Lora with 8-bit or 4-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )
        is_target_modules_in_base_model = False
        kwargs = {
            "r": lora_config.r,
            "num_heads": lora_config.num_heads,
            "be_scaling": lora_config.be_scaling,
            "be_dropout": lora_config.be_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_be_weights": lora_config.init_be_weights,
        }
        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            if isinstance(lora_config.target_modules, str):
                target_module_found = re.fullmatch(lora_config.target_modules, key)
            else:
                target_module_found = any(key.endswith(target_key) for target_key in lora_config.target_modules)
            if target_module_found:
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = _get_submodules(self.model, key)
                if hasattr(target, "bias"):
                    bias = target.bias is not None

                if isinstance(target, FLoraLayer):
                    target.update_layer(
                        adapter_name,
                        lora_config.r,
                        lora_config.num_heads,
                        lora_config.fix_alpha,
                        lora_config.lora_alpha,
                        lora_config.lora_dropout,
                        lora_config.init_lora_weights,
                    )
                else:
                    if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
                        eightbit_kwargs = kwargs.copy()
                        eightbit_kwargs.update(
                            {
                                "has_fp16_weights": target.state.has_fp16_weights,
                                "memory_efficient_backward": target.state.memory_efficient_backward,
                                "threshold": target.state.threshold,
                                "index": target.index,
                            }
                        )
                        new_module = Linear8bitLt(
                            adapter_name, target.in_features, target.out_features,
                            rank=lora_config.r, fix_alpha=lora_config.fix_alpha,
                            bias=bias, **eightbit_kwargs
                        )
                    elif loaded_in_4bit and is_bnb_4bit_available() and isinstance(target, bnb.nn.Linear4bit):
                        raise ValueError("FLora does not support 4-bit quantization yet.")
                        fourbit_kwargs = kwargs.copy()
                        fourbit_kwargs.update(
                            {
                                "compute_dtype": target.compute_dtype,
                                "compress_statistics": target.weight.compress_statistics,
                                "quant_type": target.weight.quant_type,
                            }
                        )
                        new_module = Linear4bit(
                            adapter_name, target.in_features, target.out_features,
                            rank=lora_config.r, bias=bias, **fourbit_kwargs
                        )
                    elif isinstance(target, torch.nn.Embedding):
                        raise ValueError("FLora does not support torch.nn.Embedding yet.")
                        embedding_kwargs = kwargs.copy()
                        embedding_kwargs.pop("fan_in_fan_out", None)
                        in_features, out_features = target.num_embeddings, target.embedding_dim
                        new_module = Embedding(adapter_name, in_features, out_features, **embedding_kwargs)
                    else:
                        if isinstance(target, torch.nn.Linear):
                            in_features, out_features = target.in_features, target.out_features
                            if kwargs["fan_in_fan_out"]:
                                warnings.warn(
                                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                                    "Setting fan_in_fan_out to False."
                                )
                                kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
                        elif isinstance(target, Conv1D):
                            in_features, out_features = (
                                target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
                            )
                            if not kwargs["fan_in_fan_out"]:
                                warnings.warn(
                                    "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                                    "Setting fan_in_fan_out to True."
                                )
                                kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
                        else:
                            import ipdb; ipdb.set_trace()
                            raise ValueError(
                                f"Target module {target} is not supported. "
                                f"Currently, only `torch.nn.Linear` and `Conv1D` are supported."
                            )
                        new_module = Linear(
                            adapter_name, in_features, out_features,
                            rank=lora_config.r, fix_alpha=lora_config.fix_alpha,
                            bias=bias, **kwargs)

                    self._replace_module(parent, target_name, new_module, target)
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {lora_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight
        if hasattr(old_module, "bias"):
            if old_module.bias is not None:
                new_module.bias = old_module.bias

        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "be_" in name:
                module.to(old_module.weight.device)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    def get_peft_config_as_dict(self, inference: bool = False):
        config_dict = {}
        for key, value in self.peft_config.items():
            config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(value).items()}
            if inference:
                config["inference_mode"] = True
        config_dict[key] = config
        return config

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, FLoraLayer):
                module.disable_adapters = False if enabled else True

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)

    def set_adapter(self, adapter_name):
        for module in self.model.modules():
            if isinstance(module, FLoraLayer):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    module.unmerge()
                module.active_adapter = adapter_name

    def merge_adapter(self):
        for module in self.model.modules():
            if isinstance(module, FLoraLayer):
                module.merge()

    def unmerge_adapter(self):
        for module in self.model.modules():
            if isinstance(module, FLoraLayer):
                module.unmerge()

    @staticmethod
    def _prepare_lora_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_config["model_type"]]
        if peft_config.inference_mode:
            peft_config.merge_weights = True
        return peft_config

    def merge_and_unload(self):
        r"""
        This method merges the LoRa layers into the base model. This is needed if someone wants to use the base model
        as a standalone model.
        """
        if getattr(self.config, "model_type", None) == "gpt2":
            raise ValueError("GPT2 models are not supported for merging LORA layers")

        if getattr(self.model, "is_loaded_in_8bit", False) or getattr(self.model, "is_loaded_in_4bit", False):
            raise ValueError("Cannot merge LORA layers when the model is loaded in 8-bit mode")

        key_list = [key for key, _ in self.model.named_modules() if "lora" not in key]
        for key in key_list:
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            if isinstance(target, FLoraLayer):
                bias = target.bias is not None
                new_module = torch.nn.Linear(target.in_features, target.out_features, bias=bias)
                target.merge()
                self._replace_module(parent, target_name, new_module, target)

            # save any additional trainable modules part of `modules_to_save`
            if isinstance(target, ModulesToSaveWrapper):
                setattr(parent, target_name, target.modules_to_save[target.active_adapter])

        return self.model

    def add_weighted_adapter(self, adapters, weights, adapter_name):
        if len({self.peft_config[adapter].r for adapter in adapters}) != 1:
            raise ValueError("All adapters must have the same r value")
        self.peft_config[adapter_name] = self.peft_config[adapters[0]]
        self.peft_config[adapter_name].lora_alpha = self.peft_config[adapters[0]].r
        self._find_and_replace(adapter_name)
        mark_only_be_as_trainable(self.model, self.peft_config[adapter_name].bias)
        _freeze_adapter(self.model, adapter_name)
        key_list = [key for key, _ in self.model.named_modules() if "lora" not in key]
        for key in key_list:
            _, target, _ = _get_submodules(self.model, key)
            if isinstance(target, FLoraLayer):
                if adapter_name in target.lora_A:
                    target.lora_A[adapter_name].weight.data = target.lora_A[adapter_name].weight.data * 0.0
                    target.lora_B[adapter_name].weight.data = target.lora_B[adapter_name].weight.data * 0.0
                    for adapter, weight in zip(adapters, weights):
                        if adapter not in target.lora_A:
                            continue
                        target.lora_A[adapter_name].weight.data += (
                            target.lora_A[adapter].weight.data * weight * target.scaling[adapter]
                        )
                        target.lora_B[adapter_name].weight.data += target.lora_B[adapter].weight.data * weight

                elif adapter_name in target.lora_embedding_A:
                    target.lora_embedding_A[adapter_name].data = target.lora_embedding_A[adapter_name].data * 0.0
                    target.lora_embedding_B[adapter_name].data = target.lora_embedding_B[adapter_name].data * 0.0
                    for adapter, weight in zip(adapters, weights):
                        if adapter not in target.lora_embedding_A:
                            continue
                        target.lora_embedding_A[adapter_name].data += (
                            target.lora_embedding_A[adapter].data * weight * target.scaling[adapter]
                        )
                        target.lora_embedding_B[adapter_name].data += target.lora_embedding_B[adapter].data * weight


# Below code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


# had to adapt it for `lora_only` to work
def mark_only_be_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        if "be_" not in n:
            p.requires_grad = False
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "be_only":
        for m in model.modules():
            if isinstance(m, FLoraLayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


class ElementWiseMultiply(nn.Module):
    __constants__ = ['dim']
    dim: int
    weight: torch.Tensor

    def __init__(self, size, rank, forward_type="alpha", fix_alpha=False, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        # Initialize the weights wiElementWiseMultiplyth size 'size'
        if rank > 1:
            if forward_type == "alpha":
                requires_grad = not fix_alpha
                self.weight = Parameter(
                    torch.ones((rank, ) + size, **factory_kwargs),
                    requires_grad=requires_grad)
            elif forward_type == "gamma":
                self.weight = Parameter(
                    torch.ones((rank, ) + size, **factory_kwargs),
                    requires_grad=True)
        else:
            if forward_type == "alpha":
                requires_grad = not fix_alpha
                self.weight = Parameter(
                    torch.ones(size, **factory_kwargs), requires_grad=requires_grad)
            elif forward_type == "gamma":
                self.weight = Parameter(
                    torch.ones(size, **factory_kwargs), requires_grad=True)
        self.forward_type = forward_type
        self.num_heads = size[0]
        self.dim = size[-1]
        self.rank = rank
        self.fix_alpha = fix_alpha

    def forward(self, x, **kwargs):
        if self.forward_type == "alpha":
            examples_per_model = x.shape[0] // self.weight.shape[-2]
            lang = kwargs.get("lang", None)
            if lang is None:
                if examples_per_model > 0:
                    if len(self.weight.shape) == 2:
                        alpha = self.weight.repeat([examples_per_model, 1])
                    elif len(self.weight.shape) == 3:
                        alpha = self.weight.repeat([1, examples_per_model, 1])
                    else:
                        raise ValueError("alpha has 4 dim, not supported")
                else:
                    if self.rank > 1:
                        alpha = self.weight[:,0,:].unsqueeze(1)
                    else:
                        alpha = self.weight[0].unsqueeze(0)
            else:
                alpha = torch.index_select(
                    self.weight,
                    dim=1 if self.rank > 1 else 0,
                    index=lang.to(self.weight.device).flatten()
                )
            if self.rank > 1:
                x = x.unsqueeze(0)
                alpha = alpha.unsqueeze(-2)
                result = x * alpha
                return result.reshape((-1,) + result.size()[2:])
            else:
                alpha = alpha.unsqueeze(1)
                return x * alpha
        elif self.forward_type == "gamma":
            # TODO: Implement self.scaling
            examples_per_model = x.shape[0] // self.rank // self.weight.shape[-2]
            lang = kwargs.get("lang", None)
            if lang is None:
                if examples_per_model > 0:
                    if len(self.weight.shape) == 2:
                        gamma = self.weight.repeat([examples_per_model, 1])
                    elif len(self.weight.shape) == 3:
                        gamma = self.weight.repeat([1, examples_per_model, 1])
                    else:
                        raise ValueError("gamma has 4 dim, not supported")
                else:
                    if self.rank > 1:
                        gamma = self.weight[:,0,:].unsqueeze(1)
                    else:
                        gamma = self.weight[0].unsqueeze(0)
            else:
                gamma = torch.index_select(
                    self.weight,
                    dim=1 if self.rank > 1 else 0,
                    index=lang.to(self.weight.device).flatten()
                )
            if self.rank > 1:
                x = x.reshape((self.rank, -1) + x.size()[-2:])
                gamma = gamma.unsqueeze(-2)
                return torch.mean(gamma * x, dim=0)
            else:
                gamma = gamma.unsqueeze(1)
                return gamma * x
        else:
            raise ValueError(
                f"self.forward_type must be either alpha or gamma, got {self.forward_type}")

    def __repr__(self) -> str:
        return f"ElementWiseMultiply(num_heads={self.num_heads}, dim={self.dim}, rank={self.rank}, fix_alpha={self.fix_alpha}, forward_type={self.forward_type})"

    def randomize(self):
        with torch.no_grad():
            self.weight.uniform_(0.0, 1.0)


class FLoraLayer:
    def __init__(
        self,
        in_features: int,
        out_features: int,
    ):
        self.r = {}
        self.be_scaling = {}
        self.scaling = {}
        self.be_dropout = nn.ModuleDict({})
        self.be_alpha = nn.ModuleDict({})
        self.be_gamma = nn.ModuleDict({})
        #self.be_alpha = {}
        #self.be_gamma = {}
        # For Embedding layer
        #self.lora_embedding_A = nn.ParameterDict({})
        #self.lora_embedding_B = nn.ParameterDict({})
        # Mark the weight as unmerged
        self.merged = False
        self.disable_adapters = False
        self.in_features = in_features
        self.out_features = out_features

    def update_layer(self, adapter_name, r, num_heads, fix_alpha, be_scaling, be_dropout, init_be_weights):
        self.r[adapter_name] = r
        self.be_scaling[adapter_name] = be_scaling
        self.num_heads = num_heads
        if be_dropout > 0.0:
            be_dropout_layer = nn.Dropout(p=be_dropout)
        else:
            be_dropout_layer = nn.Identity()

        self.be_dropout.update(nn.ModuleDict({adapter_name: be_dropout_layer}))
        # Actual trainable parameters
        if r > 0:
            self.be_alpha.update(
                nn.ModuleDict({adapter_name: ElementWiseMultiply(
                    (num_heads, self.in_features), r, "alpha", fix_alpha)}))
            self.be_gamma.update(
                nn.ModuleDict({adapter_name: ElementWiseMultiply(
                    (num_heads, self.out_features), r, "gamma")}))
            #self.be_alpha.update(
            #    {adapter_name: nn.Parameter(
            #    torch.ones(num_heads, self.in_features) if r == 1 else torch.ones(r, num_heads, self.in_features))}
            #)
            #self.be_gamma.update(
            #    {adapter_name: nn.Parameter(
            #    torch.zeros(num_heads, self.out_features) if r == 1 else torch.ones(r, num_heads, self.out_features))}
            #)
            #self.scaling[adapter_name] = be_alpha / r
            # Use scaling 1.0 in BE for now.
            self.scaling[adapter_name] = 1.0
        if init_be_weights:
            self.reset_be_parameters(adapter_name)
        self.to(self.weight.device)

    def update_layer_embedding(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        raise ValueError("Not implemented")
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        if r > 0:
            self.lora_embedding_A.update(
                nn.ParameterDict({adapter_name: nn.Parameter(self.weight.new_zeros((r, self.in_features)))})
            )
            self.lora_embedding_B.update(
                nn.ParameterDict({adapter_name: nn.Parameter(self.weight.new_zeros((self.out_features, r)))})
            )
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)

    def reset_be_parameters(self, adapter_name):
        if adapter_name in self.be_alpha.keys():
            nn.init.ones_(self.be_alpha[adapter_name].weight)
        if adapter_name in self.be_gamma.keys():
            # initialize A the same way as the default for nn.Linear and B to zero
            # Initialize the parameters to be centered around zero, need to add 1 in the forward.
            nn.init.ones_(self.be_gamma[adapter_name].weight)
            #nn.init.kaiming_uniform_(self.be_gamma[adapter_name].weight, a=math.sqrt(5))
        #if adapter_name in self.lora_embedding_A.keys():
        #    # initialize a the same way as the default for nn.linear and b to zero
        #    nn.init.zeros_(self.lora_embedding_A[adapter_name])
        #    nn.init.normal_(self.lora_embedding_B[adapter_name])


class Linear(nn.Linear, FLoraLayer):
    # FLora implemented in a dense layer
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        rank: int = 0,
        num_heads: int = 1,
        fix_alpha: bool = False,
        be_scaling: int = 1,
        be_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        **kwargs,
    ):
        init_be_weights = kwargs.pop("init_be_weights", True)
        _ = kwargs.pop("r", 1)

        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        FLoraLayer.__init__(self, in_features=in_features, out_features=out_features)
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False
        self.num_heads = num_heads

        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        nn.Linear.reset_parameters(self)
        self.rank = rank
        self.update_layer(adapter_name, rank, num_heads, fix_alpha, be_scaling, be_dropout, init_be_weights)
        self.active_adapter = adapter_name

    def merge(self):
        if self.active_adapter not in self.lora_A.keys():
            return
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.weight.data += (
                transpose(
                    self.lora_B[self.active_adapter].weight @ self.lora_A[self.active_adapter].weight,
                    self.fan_in_fan_out,
                )
                * self.scaling[self.active_adapter]
            )
            self.merged = True

    def unmerge(self):
        if self.active_adapter not in self.lora_A.keys():
            return
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.weight.data -= (
                transpose(
                    self.lora_B[self.active_adapter].weight @ self.lora_A[self.active_adapter].weight,
                    self.fan_in_fan_out,
                )
                * self.scaling[self.active_adapter]
            )
            self.merged = False

    def forward(self, x: torch.Tensor, **kwargs):
        previous_dtype = x.dtype
        if self.active_adapter not in self.be_alpha.keys():
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        if self.disable_adapters:
            if self.r[self.active_adapter] > 0 and self.merged:
                self.unmerge()
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        elif self.r[self.active_adapter] > 0 and not self.merged:
            x = x.to(self.be_alpha[self.active_adapter].weight.dtype)

            rank = self.r[self.active_adapter]
            # TODO: implement dropout.
            # TODO: check the which dimension is seq_length and which is hidden_size.
            x = self.be_alpha[self.active_adapter](x, **kwargs)
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out))
            # Not sure whether need to group bias to different tasks.
            result = self.be_gamma[self.active_adapter](result, **kwargs) + self.bias

            #x = x.view(self.num_heads, examples_per_model, -1, self.in_features)
            #x = self.be_alpha[self.active_adapter](x)
            #x = x.reshape([batch_size, -1, self.in_features])
            #result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            #result = result.view(self.num_heads, examples_per_model, -1, self.out_features)
            #result = self.be_gamma[self.active_adapter](result)
            #result = result.reshape([batch_size, -1, self.out_features])
        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        result = result.to(previous_dtype)

        return result


if is_bnb_available():

    class Linear8bitLt(bnb.nn.Linear8bitLt, FLoraLayer):
        # Lora implemented in a dense layer
        def __init__(
            self,
            adapter_name,
            in_features,
            out_features,
            rank: int = 0,
            num_heads: int = 1,
            fix_alpha: bool = False,
            be_scaling: int = 1,
            be_dropout: float = 0.0,
            **kwargs,
        ):
            bnb.nn.Linear8bitLt.__init__(
                self,
                in_features,
                out_features,
                bias=kwargs.get("bias", True),
                has_fp16_weights=kwargs.get("has_fp16_weights", True),
                memory_efficient_backward=kwargs.get("memory_efficient_backward", False),
                threshold=kwargs.get("threshold", 0.0),
                index=kwargs.get("index", None),
            )
            FLoraLayer.__init__(self, in_features=in_features, out_features=out_features)

            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            init_be_weights = kwargs.pop("init_be_weights", True)
            self.rank = rank
            self.update_layer(adapter_name, rank, num_heads, fix_alpha, be_scaling, be_dropout, init_be_weights)
            self.active_adapter = adapter_name

        def forward(self, x: torch.Tensor, **kwargs):
            if self.disable_adapters or self.active_adapter not in self.be_alpha.keys():
                result = super().forward(x)
                return result
            elif self.r[self.active_adapter] > 0:
                rank = self.r[self.active_adapter]
                if not torch.is_autocast_enabled():
                    expected_dtype = x.dtype

                    if x.dtype != torch.float32:
                        x = x.float()
                    x = self.be_alpha[self.active_adapter](x, **kwargs)

                    self.state.is_training = self.training
                    if self.weight.CB is not None:
                        self.init_8bit_state()

                    out = bnb.matmul(x, self.weight, state=self.state)
                    if not self.state.has_fp16_weights:
                        if self.state.CB is not None and self.state.CxB is not None:
                            # we converted 8-bit row major to turing/ampere format in the first inference pass
                            # we no longer need the row-major weight
                            del self.state.CB
                            self.weight.data = self.state.CxB

                    # Not sure whether need to group bias to different tasks.
                    if self.bias is not None:
                        result = (self.be_gamma[self.active_adapter](
                            out, **kwargs) + self.bias).to(expected_dtype)
                    else:
                        result = self.be_gamma[self.active_adapter](
                            out, **kwargs).to(expected_dtype)

                    #x = x.view(self.num_heads, examples_per_model, -1, self.in_features)
                    #x = self.be_alpha[self.active_adapter](x)
                    #x = x.reshape([batch_size, -1, self.in_features])
                    #result = super().forward(x)
                    #result = result.view(self.num_heads, examples_per_model, -1, self.out_features)
                    #result = self.be_gamma[self.active_adapter](result)
                    #result = result.reshape([batch_size, -1, self.out_features]).to(expected_dtype)
                else:
                    x = self.be_alpha[self.active_adapter](x, **kwargs)

                    #result = super().forward(x)
                    self.state.is_training = self.training
                    if self.weight.CB is not None:
                        self.init_8bit_state()

                    out = bnb.matmul(x, self.weight, state=self.state)
                    if not self.state.has_fp16_weights:
                        if self.state.CB is not None and self.state.CxB is not None:
                            # we converted 8-bit row major to turing/ampere format in the first inference pass
                            # we no longer need the row-major weight
                            del self.state.CB
                            self.weight.data = self.state.CxB

                    # Not sure whether need to group bias to different tasks.
                    # weights are cast automatically as Int8Params, but the bias has to be cast manually
                    if self.bias is not None and self.bias.dtype != x.dtype:
                        self.bias.data = self.bias.data.to(x.dtype)
                    result = self.be_gamma[self.active_adapter](out, **kwargs)
                    if self.bias is not None:
                        result += self.bias

                    #self.be_alpha["default"].randomize()
                    #self.be_gamma["default"].randomize()
                    #tt1 = self.be_alpha[self.active_adapter](x)
                    #my_weight = torch.randn(self.weight.shape).to(self.weight.device)
                    #tt2 = F.linear(tt1, my_weight)
                    #tt2 = super().forward(tt1)
                    #tt3 = self.be_gamma[self.active_adapter](tt2)

                    #kk = torch.bmm(self.be_gamma["default"].weight.unsqueeze(2),self.be_alpha["default"].weight.unsqueeze(1))
                    #new_weight = my_weight * kk[1]

                    #import ipdb;ipdb.set_trace()

                    #x = x.view(self.num_heads, examples_per_model, -1, self.in_features)
                    #x = self.be_alpha[self.active_adapter](x)
                    #x = x.reshape([batch_size, -1, self.in_features])
                    #result = super().forward(x)
                    #result = result.view(self.num_heads, examples_per_model, -1, self.out_features)
                    #result = self.be_gamma[self.active_adapter](result)
                    #result = result.reshape([batch_size, -1, self.out_features])
            return result


    if is_bnb_4bit_available():

        class Linear4bit(bnb.nn.Linear4bit, FLoraLayer):
            # Lora implemented in a dense layer
            def __init__(
                self,
                adapter_name,
                in_features,
                out_features,
                r: int = 0,
                lora_alpha: int = 1,
                lora_dropout: float = 0.0,
                **kwargs,
            ):
                bnb.nn.Linear4bit.__init__(
                    self,
                    in_features,
                    out_features,
                    bias=kwargs.get("bias", True),
                    compute_dtype=kwargs.get("compute_dtype", torch.float32),
                    compress_statistics=kwargs.get("compress_statistics", True),
                    quant_type=kwargs.get("quant_type", "nf4"),
                )
                FLoraLayer.__init__(self, in_features=in_features, out_features=out_features)

                # Freezing the pre-trained weight matrix
                self.weight.requires_grad = False

                init_lora_weights = kwargs.pop("init_lora_weights", True)
                self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
                self.active_adapter = adapter_name

            def forward(self, x: torch.Tensor, **kwargs):
                result = super().forward(x)

                if self.disable_adapters or self.active_adapter not in self.lora_A.keys():
                    return result
                elif self.r[self.active_adapter] > 0:
                    result = result.clone()
                    if not torch.is_autocast_enabled():
                        expected_dtype = result.dtype
                        x = x.to(self.lora_A[self.active_adapter].weight.dtype)
                        output = (
                            self.lora_B[self.active_adapter](
                                self.lora_A[self.active_adapter](self.lora_dropout[self.active_adapter](x))
                            ).to(expected_dtype)
                            * self.scaling[self.active_adapter]
                        )
                    else:
                        output = (
                            self.lora_B[self.active_adapter](
                                self.lora_A[self.active_adapter](self.lora_dropout[self.active_adapter](x))
                            )
                            * self.scaling[self.active_adapter]
                        )
                    result += output
                return result
