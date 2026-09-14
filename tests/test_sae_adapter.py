"""CPU checks for Senna's raw-to-expanded activation mapping."""

from types import SimpleNamespace

import torch
from torch import nn

from llava.constants import IMAGE_TOKEN_INDEX
from llava.senna.sae_adapter import SennaActivationAdapter


class _FakeDecoderLayer(nn.Module):
    def forward(self, hidden_states, **kwargs):
        return hidden_states + 2, None


class _FakeSenna(nn.Module):
    def __init__(self, hidden_size=4, visual_tokens=3):
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=hidden_size,
            tokenizer_padding_side="right",
            tokenizer_model_max_length=None,
            _name_or_path="fake-senna",
        )
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_FakeDecoderLayer()])
        self.model.img_adapter = SimpleNamespace(new_token_num=visual_tokens)
        self.projection = nn.Linear(hidden_size, 5, bias=False)

    def _prepare(self, input_ids, attention_mask, images):
        mask = torch.ones_like(input_ids, dtype=torch.bool) if attention_mask is None else attention_mask.bool()
        rows = []
        masks = []
        for ids, valid in zip(input_ids, mask):
            ids = ids[valid]
            pieces = []
            row_mask = []
            for token in ids.tolist():
                if token == IMAGE_TOKEN_INDEX:
                    pieces.append(torch.ones(3, 4))
                    row_mask.extend([True] * 3)
                else:
                    pieces.append(torch.full((1, 4), float(token)))
                    row_mask.append(True)
            rows.append(torch.cat(pieces))
            masks.append(torch.tensor(row_mask))
        width = max(row.shape[0] for row in rows)
        padded = []
        expanded_mask = torch.zeros((len(rows), width), dtype=torch.bool)
        for row_idx, row in enumerate(rows):
            padded.append(torch.cat((row, torch.zeros((width - row.shape[0], 4)))))
            expanded_mask[row_idx, : row.shape[0]] = True
        return None, None, expanded_mask, None, torch.stack(padded), None

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, images, image_sizes
    ):
        return self._prepare(input_ids, attention_mask, images)

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, **kwargs):
        if inputs_embeds is None:
            prepared = self._prepare(input_ids, attention_mask, kwargs.get("images"))
            inputs_embeds = prepared[4]
            attention_mask = prepared[2]
        hidden = inputs_embeds
        hidden = self.model.layers[0](hidden, attention_mask=attention_mask)[0]
        return SimpleNamespace(logits=self.projection(hidden))


def test_senna_capture_uses_expanded_positions_and_preserves_logits():
    model = _FakeSenna()
    adapter = SennaActivationAdapter(
        model,
        layer=0,
        checkpoint="fake-senna",
        tokenizer="fake-tokenizer",
        prompt_template="fake-template",
    )
    input_ids = torch.tensor(
        [[7, IMAGE_TOKEN_INDEX, 8, 9, 0], [0, 5, IMAGE_TOKEN_INDEX, 6, 0]],
        dtype=torch.long,
    )
    attention_mask = torch.tensor([[1, 1, 1, 1, 0], [0, 1, 1, 1, 0]])
    images = torch.zeros((2, 1, 3, 2, 2))

    native = model(input_ids=input_ids, attention_mask=attention_mask, images=images)
    captured = adapter.forward(
        input_ids,
        attention_mask=attention_mask,
        images=images,
    )
    first = adapter.forward(
        input_ids[:1, :4],
        attention_mask=attention_mask[:1, :4],
        images=images[:1],
    )
    second = adapter.forward(
        input_ids[1:],
        attention_mask=attention_mask[1:],
        images=images[1:],
    )

    torch.testing.assert_close(captured.output.logits, native.logits)
    torch.testing.assert_close(
        captured.activations,
        torch.cat((first.activations, second.activations)),
    )
    assert captured.activations.shape == (5, 4)
    assert captured.metadata["selected_sequence_indices"][0].tolist() == [0, 4, 5]
    assert captured.metadata["selected_raw_token_indices"][0].tolist() == [0, 2, 3]
    assert captured.metadata["selected_sequence_indices"][1].tolist() == [0, 4]
    assert captured.metadata["selected_raw_token_indices"][1].tolist() == [1, 3]
    assert captured.metadata["final_prompt_positions"].tolist() == [5, 4]
    assert captured.metadata["module_path"] == "model.model.layers[0]"
    assert captured.metadata["capture_point"] == "after_complete_decoder_layer_forward"
