import unittest

import numpy as np
import pandas as pd
from fastapi import HTTPException

from app.services.ml import build_pipeline, train_model
from app.services.preprocessing import run_preprocessing
from app.main import _profile


class PreprocessingTests(unittest.TestCase):
    def test_exact_duplicates_are_removed_and_categories_stay_readable(self):
        raw = pd.DataFrame(
            {
                "color": ["red", "red", "blue", "green"],
                "measurement": [1.0, 1.0, 2.0, 3.0],
                "target": ["yes", "yes", "no", "yes"],
            }
        )

        cleaned, report = run_preprocessing(
            raw,
            target="target",
            imputation="simple",
            outlier_treatment="none",
            drop_duplicates=True,
        )

        self.assertEqual(len(cleaned), 3)
        self.assertEqual(report["summary"]["duplicates_removed"], 1)
        self.assertEqual(cleaned["color"].tolist(), ["red", "blue", "green"])
        self.assertEqual(report["steps"][2]["method"], "Rule-based")

    def test_model_pipeline_one_hot_encodes_categorical_features(self):
        features = pd.DataFrame(
            {
                "measurement": [1.0, 2.0, 3.0, 4.0],
                "color": ["red", "blue", "red", "green"],
            }
        )
        pipeline = build_pipeline(
            num_cols=["measurement"],
            cat_cols=["color"],
            task="classification",
            params={"n_estimators": 10},
        )
        pipeline.fit(features, pd.Series([0, 1, 0, 1]))

        encoder = pipeline.named_steps["prep"].named_transformers_["cat_onehot"].named_steps["oh"]
        self.assertEqual(set(encoder.categories_[0]), {"blue", "green", "red"})
        self.assertEqual(pipeline.named_steps["prep"].transform(features).shape[1], 4)

    def test_model_pipeline_supports_label_and_sorted_ordinal_encoding(self):
        features = pd.DataFrame({"size": ["small", "medium", "large", "small"]})
        target = pd.Series([0, 1, 0, 1])
        pipeline = build_pipeline(
            num_cols=[],
            cat_cols=["size"],
            task="classification",
            params={"n_estimators": 10},
            categorical_encoding="ordinal",
        )
        pipeline.fit(features, target)

        encoded = pipeline.named_steps["prep"].transform(features)
        self.assertEqual(encoded[:, 0].tolist(), [2.0, 1.0, 0.0, 2.0])
        unknown = pipeline.named_steps["prep"].transform(pd.DataFrame({"size": ["extra"]}))
        self.assertEqual(unknown[0, 0], -1.0)

        label_pipeline = build_pipeline(
            num_cols=[],
            cat_cols=["size"],
            task="classification",
            params={"n_estimators": 10},
            categorical_encoding="label",
        )
        label_pipeline.fit(features, target)
        self.assertEqual(
            label_pipeline.named_steps["prep"].transform(features)[:, 0].tolist(),
            [0.0, 1.0, 2.0, 0.0],
        )

    def test_training_with_global_ordinal_encoding_reports_original_feature(self):
        data = pd.DataFrame({
            "size": ["small", "medium", "large"] * 20,
            "target": ["no", "yes"] * 30,
        })

        _, results, _ = train_model(
            data,
            "target",
            n_estimators=10,
            categorical_encoding="ordinal",
        )

        self.assertIn("size", [item["feature"] for item in results["feature_importance"]])
        self.assertEqual(results["categorical_encoding"], "ordinal")

    def test_training_rejects_unknown_global_encoding(self):
        data = pd.DataFrame({
            "category": ["low", "high"] * 15,
            "target": ["yes", "no"] * 15,
        })

        with self.assertRaises(HTTPException) as error:
            train_model(
                data,
                "target",
                n_estimators=10,
                categorical_encoding="unknown",
            )

        self.assertEqual(error.exception.status_code, 400)
        self.assertIn("Unsupported categorical encoding", error.exception.detail)

    def test_training_supports_selected_classification_and_regression_models(self):
        classification = pd.DataFrame({
            "feature": np.arange(60, dtype=float),
            "target": ["yes", "no"] * 30,
        })
        _, classification_results, _ = train_model(
            classification,
            "target",
            model_name="logistic_regression",
        )
        self.assertEqual(classification_results["task"], "classification")
        self.assertEqual(classification_results["model"], "logistic_regression")

        regression = pd.DataFrame({
            "feature": np.arange(60, dtype=float),
            "target": np.arange(60, dtype=float) * 2 + 0.5,
        })
        _, regression_results, _ = train_model(
            regression,
            "target",
            model_name="ridge",
        )
        self.assertEqual(regression_results["task"], "regression")
        self.assertEqual(regression_results["model"], "ridge")

    def test_mean_median_and_mode_imputation_options(self):
        data = pd.DataFrame({
            "measurement": [1.0, 1.0, np.nan, 5.0, 9.0],
            "category": ["a", "b", "a", None, "b"],
        })
        expected = {"mean": 4.0, "median": 3.0, "most_frequent": 1.0}

        for method, numeric_fill in expected.items():
            with self.subTest(method=method):
                clean, report = run_preprocessing(
                    data,
                    imputation=method,
                    outlier_treatment="none",
                    drop_duplicates=False,
                )
                self.assertAlmostEqual(clean.loc[2, "measurement"], numeric_fill)
                self.assertEqual(clean.loc[3, "category"], "a")
                self.assertEqual(report["summary"]["cells_imputed"], 2)

    def test_global_encoding_option_is_saved_and_cleaned_categories_stay_readable(self):
        data = pd.DataFrame({"grade": ["low", "LOW", "medium", "HIGH"]})

        clean, report = run_preprocessing(
            data,
            outlier_treatment="none",
            drop_duplicates=False,
            categorical_encoding="ordinal",
        )

        self.assertEqual(report["options"]["categorical_encoding"], "ordinal")
        self.assertTrue(all(isinstance(value, str) for value in clean["grade"]))
        self.assertEqual(clean["grade"].nunique(), 3)

    def test_profile_recommends_classification_or_regression_from_target_values(self):
        frame = pd.DataFrame(
            {
                "class_target": ["yes", "no"] * 20,
                "numeric_target": [float(value) + 0.25 for value in range(40)],
                "single_class": ["only"] * 40,
            }
        )

        profile = {column["name"]: column for column in _profile(frame)}

        self.assertEqual(profile["class_target"]["prediction_task"], "classification")
        self.assertTrue(profile["class_target"]["prediction_supported"])
        self.assertEqual(profile["numeric_target"]["prediction_task"], "regression")
        self.assertTrue(profile["numeric_target"]["prediction_supported"])
        self.assertFalse(profile["single_class"]["prediction_supported"])


if __name__ == "__main__":
    unittest.main()
