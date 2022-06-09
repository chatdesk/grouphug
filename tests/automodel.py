from grouphug import AutoMultiTaskModel, ClassificationHeadConfig, LMHeadConfig


def test_automodel():
    base_model = "vinai/bertweet-base"
    head_configs = [
        LMHeadConfig(),
        ClassificationHeadConfig(problem_type="multi_label_classification", name="topics", num_labels=10),
        ClassificationHeadConfig(
            problem_type="single_label_classification",
            num_labels=3,
            labels_var="action_label",
        ),
    ]
    model = AutoMultiTaskModel.from_pretrained(base_model, head_configs)
