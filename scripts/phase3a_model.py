"""Model helpers for the Phase 3A coverage-auxiliary Controller."""

from __future__ import annotations

from phase2_model import load_binary_sequence_classifier


def add_coverage_head(sequence_classifier, torch_module):
    """Wrap a binary sequence classifier with a bounded coverage-regression head."""

    class CoverageAuxiliaryModel(torch_module.nn.Module):
        def __init__(self, classifier):
            super().__init__()
            self.sequence_classifier = classifier
            hidden_size = int(classifier.config.hidden_size)
            self.coverage_head = torch_module.nn.Linear(hidden_size, 1)
            classifier._init_weights(self.coverage_head)

        @property
        def config(self):
            return self.sequence_classifier.config

        def forward(self, **encoded):
            backbone = getattr(
                self.sequence_classifier,
                self.sequence_classifier.base_model_prefix,
            )
            outputs = backbone(**encoded, return_dict=True)
            sequence_output = outputs.last_hidden_state
            stop_logits = self.sequence_classifier.classifier(sequence_output)
            coverage = torch_module.sigmoid(
                self.coverage_head(sequence_output[:, 0, :])
            ).squeeze(-1)
            return stop_logits, coverage

    return CoverageAuxiliaryModel(sequence_classifier)


def load_phase3a_model(backbone: str, load_kwargs: dict, coverage_auxiliary: bool):
    import torch

    classifier, initialization = load_binary_sequence_classifier(backbone, load_kwargs)
    if not coverage_auxiliary:
        return classifier, {**initialization, "coverage_head": "disabled"}
    model = add_coverage_head(classifier, torch)
    return model, {
        **initialization,
        "coverage_head": "linear_sigmoid_from_shared_cls_backbone_state",
        "coverage_output_range": [0.0, 1.0],
    }
