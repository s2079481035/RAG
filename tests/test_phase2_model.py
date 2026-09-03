import unittest

from scripts.phase2_model import configure_cublas_workspace, replace_with_binary_output_head


class FakeLinear:
    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.bias = object() if bias else None
        self.initialized = False


class FakeTorch:
    class nn:
        Linear = FakeLinear


class FakeConfig:
    num_labels = 1
    problem_type = "regression"
    id2label = {0: "LABEL_0"}
    label2id = {"LABEL_0": 0}


class FakeClassifier:
    out_proj = FakeLinear(768, 1)


class FakeModel:
    def __init__(self):
        self.classifier = FakeClassifier()
        self.config = FakeConfig()
        self.num_labels = 1

    @staticmethod
    def _init_weights(layer):
        layer.initialized = True


class Phase2ModelTests(unittest.TestCase):
    def test_cublas_workspace_is_set_before_torch_use(self):
        environment = {}

        value = configure_cublas_workspace(":4096:8", environment)

        self.assertEqual(value, ":4096:8")
        self.assertEqual(environment["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")

    def test_conflicting_cublas_workspace_is_rejected(self):
        with self.assertRaises(ValueError):
            configure_cublas_workspace(":4096:8", {"CUBLAS_WORKSPACE_CONFIG": ":16:8"})

    def test_scalar_reranker_head_is_reinitialized_for_binary_controller(self):
        model = FakeModel()

        audit = replace_with_binary_output_head(model, FakeTorch)

        self.assertEqual(audit["checkpoint_num_labels"], 1)
        self.assertEqual(model.classifier.out_proj.out_features, 2)
        self.assertTrue(model.classifier.out_proj.initialized)
        self.assertEqual(model.num_labels, 2)
        self.assertIsNone(model.config.problem_type)
        self.assertEqual(model.config.id2label, {0: "continue", 1: "stop"})


if __name__ == "__main__":
    unittest.main()
