"""Activation capture for Senna's native multimodal LLaVA forward path."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from llava.constants import IMAGE_TOKEN_INDEX


@dataclass
class SennaActivationCapture:
    """Text residual activations and the untouched native model output."""

    output: Any
    activations: torch.Tensor
    metadata: Dict[str, Any]


class SennaActivationAdapter:
    """Capture ``model.model.layers[layer]`` without changing model behavior.

    Images must already be processed by Senna's evaluation preprocessor.  The
    adapter calls ``prepare_inputs_labels_for_multimodal`` itself so it can
    retain the expanded attention mask and map raw text positions through the
    image-sentinel expansion.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        layer: int,
        *,
        checkpoint: Optional[str] = None,
        tokenizer: Any = None,
        prompt_template: Optional[str] = None,
    ) -> None:
        self.model = model
        self.layer = int(layer)
        self.checkpoint = checkpoint
        self.tokenizer = tokenizer
        self.prompt_template = prompt_template
        try:
            self.module = self.model.model.layers[self.layer]
        except (AttributeError, IndexError, TypeError) as exc:
            raise ValueError(
                "SennaActivationAdapter requires model.model.layers and a "
                f"valid layer index, got {self.layer}."
            ) from exc
        self.module_path = f"model.model.layers[{self.layer}]"

    @property
    def d_model(self) -> int:
        return int(self.model.config.hidden_size)

    def _visual_tokens_per_image(self) -> int:
        return int(self.model.model.img_adapter.new_token_num)

    @staticmethod
    def _layer_hidden_states(output: Any) -> torch.Tensor:
        if isinstance(output, (tuple, list)):
            output = output[0]
        if not torch.is_tensor(output):
            raise TypeError("The hooked decoder layer did not return hidden states.")
        return output

    @staticmethod
    def _raw_mask(input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if attention_mask is None:
            return torch.ones_like(input_ids, dtype=torch.bool)
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids shape.")
        return attention_mask.to(dtype=torch.bool)

    def _prepare(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        images: Optional[torch.Tensor],
        image_sizes: Optional[Sequence[Sequence[int]]],
        position_ids: Optional[torch.Tensor],
    ) -> Tuple[Tuple[Any, ...], torch.Tensor, torch.Tensor]:
        raw_mask = self._raw_mask(input_ids, attention_mask)
        prep_mask = raw_mask if attention_mask is None else attention_mask
        prepared = self.model.prepare_inputs_labels_for_multimodal(
            input_ids,
            position_ids,
            prep_mask,
            None,
            None,
            images,
            image_sizes,
        )
        expanded_mask = prepared[2]
        if expanded_mask is None:
            raise RuntimeError("Senna multimodal preparation did not return an attention mask.")
        return prepared, raw_mask, expanded_mask.to(dtype=torch.bool)

    def _text_positions(
        self,
        input_ids: torch.Tensor,
        raw_mask: torch.Tensor,
        expanded_mask: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        visual_count = self._visual_tokens_per_image()
        max_length = getattr(getattr(self.model, "config", None), "tokenizer_model_max_length", None)
        selected: List[torch.Tensor] = []
        raw_selected: List[torch.Tensor] = []
        final_positions: List[int] = []

        for batch_idx in range(input_ids.shape[0]):
            raw_positions = torch.where(raw_mask[batch_idx])[0]
            raw_tokens = input_ids[batch_idx, raw_positions]

            expanded_to_raw: List[int] = []
            for raw_idx, token in zip(raw_positions.tolist(), raw_tokens.tolist()):
                if token == IMAGE_TOKEN_INDEX:
                    expanded_to_raw.extend([-1] * visual_count)
                else:
                    expanded_to_raw.append(raw_idx)
            if max_length is not None:
                expanded_to_raw = expanded_to_raw[: int(max_length)]

            expanded_positions = torch.where(expanded_mask[batch_idx])[0]
            if len(expanded_positions) != len(expanded_to_raw):
                raise ValueError(
                    "Image-sentinel expansion does not match the model mask: "
                    f"mapping={len(expanded_to_raw)}, mask={len(expanded_positions)}."
                )
            expanded_to_raw_tensor = torch.tensor(
                expanded_to_raw, dtype=torch.long, device=input_ids.device
            )
            text_entries = expanded_to_raw_tensor >= 0
            text_positions = expanded_positions[text_entries]
            text_raw_positions = expanded_to_raw_tensor[text_entries]
            if len(text_positions) == 0:
                raise ValueError(f"Batch item {batch_idx} has no valid text position.")
            selected.append(text_positions)
            raw_selected.append(text_raw_positions)
            final_positions.append(int(text_positions[-1].item()))

        return selected, raw_selected, torch.tensor(
            final_positions, dtype=torch.long, device=input_ids.device
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[Sequence[Sequence[int]]] = None,
        position_ids: Optional[torch.Tensor] = None,
        **forward_kwargs: Any,
    ) -> SennaActivationCapture:
        """Run the native forward path and return ``(n, d_model)`` activations."""

        if input_ids is None or input_ids.ndim != 2:
            raise ValueError("input_ids must be [batch, raw_sequence_length].")
        if "inputs_embeds" in forward_kwargs:
            raise ValueError("Capture accepts input_ids and performs native multimodal preparation.")

        prepared, raw_mask, expanded_mask = self._prepare(
            input_ids,
            attention_mask,
            images,
            image_sizes,
            position_ids,
        )
        (
            prepared_input_ids,
            prepared_position_ids,
            prepared_attention_mask,
            prepared_past,
            prepared_embeds,
            prepared_labels,
        ) = prepared
        selected, raw_selected, final_positions = self._text_positions(
            input_ids, raw_mask, expanded_mask
        )

        captured: Dict[str, torch.Tensor] = {}

        def save_output(_module: torch.nn.Module, _args: Tuple[Any, ...], output: Any) -> None:
            captured["hidden_states"] = self._layer_hidden_states(output).detach()

        hook = self.module.register_forward_hook(save_output)
        try:
            if prepared_embeds is None:
                native_kwargs = dict(
                    input_ids=prepared_input_ids,
                    attention_mask=attention_mask,
                    position_ids=prepared_position_ids,
                    past_key_values=prepared_past,
                    labels=prepared_labels,
                )
                if images is not None:
                    native_kwargs["images"] = images
                if image_sizes is not None:
                    native_kwargs["image_sizes"] = image_sizes
                native_output = self.model(**native_kwargs, **forward_kwargs)
            else:
                native_output = self.model(
                    input_ids=prepared_input_ids,
                    # Keep the native mask dtype for the forward call.  The
                    # bool copy is used only for selecting text positions.
                    attention_mask=prepared_attention_mask if attention_mask is not None else None,
                    position_ids=prepared_position_ids,
                    past_key_values=prepared_past,
                    inputs_embeds=prepared_embeds,
                    labels=prepared_labels,
                    **forward_kwargs,
                )
        finally:
            hook.remove()

        hidden_states = captured.get("hidden_states")
        if hidden_states is None:
            raise RuntimeError(f"No output was captured from {self.module_path}.")
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.d_model:
            raise ValueError(
                "Expected hooked output [batch, sequence, d_model], got "
                f"{tuple(hidden_states.shape)} with d_model={self.d_model}."
            )
        activations = torch.cat(
            [hidden_states[row, positions] for row, positions in enumerate(selected)], dim=0
        ).contiguous()
        tokenizer_name = self.tokenizer
        if tokenizer_name is not None and not isinstance(tokenizer_name, str):
            tokenizer_name = getattr(tokenizer_name, "name_or_path", repr(tokenizer_name))
        checkpoint = self.checkpoint or getattr(
            getattr(self.model, "config", None), "_name_or_path", None
        )
        metadata: Dict[str, Any] = {
            "checkpoint": checkpoint,
            "tokenizer": tokenizer_name,
            "prompt_template": self.prompt_template,
            "layer": self.layer,
            "module_path": self.module_path,
            "capture_point": "after_complete_decoder_layer_forward",
            "after_final_norm": False,
            "d_model": self.d_model,
            "mask": expanded_mask.detach().cpu().clone(),
            "selected_sequence_indices": [
                value.detach().cpu().clone() for value in selected
            ],
            "selected_raw_token_indices": [
                value.detach().cpu().clone() for value in raw_selected
            ],
            "final_prompt_positions": final_positions.detach().cpu().clone(),
        }
        return SennaActivationCapture(native_output, activations, metadata)

__all__ = ["SennaActivationAdapter", "SennaActivationCapture"]
