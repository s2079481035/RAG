"""Model-loading helpers for Phase 2 binary Controllers."""

from __future__ import annotations


BINARY_LABEL_NAMES = {0: "continue", 1: "stop"}


def replace_with_binary_output_head(model, torch_module) -> dict:
    """Replace BGE's scalar reranking output with a freshly initialized binary head."""
    classifier = getattr(model, "classifier", None)
    output_layer = getattr(classifier, "out_proj", None)
    if output_layer is None or not isinstance(output_layer, torch_module.nn.Linear):
        raise TypeError(
            "Expected a sequence-classification model with classifier.out_proj Linear"
        )

    checkpoint_num_labels = int(output_layer.out_features)
    if checkpoint_num_labels != 2:
        replacement = torch_module.nn.Linear(
            int(output_layer.in_features),
            2,
            bias=output_layer.bias is not None,
        )
        model._init_weights(replacement)
        classifier.out_proj = replacement

    model.num_labels = 2
    model.config.num_labels = 2
    model.config.problem_type = None
    model.config.id2label = dict(BINARY_LABEL_NAMES)
    model.config.label2id = {label: index for index, label in BINARY_LABEL_NAMES.items()}
    return {
        "strategy": "load_checkpoint_head_then_reinitialize_binary_out_proj",
        "checkpoint_num_labels": checkpoint_num_labels,
        "resolved_num_labels": 2,
    }


def load_binary_sequence_classifier(backbone: str, load_kwargs: dict):
    """Load checkpoint-shaped weights before replacing its scalar BGE output head."""
    import torch
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(backbone, **load_kwargs)
    initialization = replace_with_binary_output_head(model, torch)
    return model, initialization
