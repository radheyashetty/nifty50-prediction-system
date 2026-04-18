"""
Explainability Module
Implements SHAP and LIME for model interpretability
"""

import numpy as np
import shap
from typing import Optional, List
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")


class ModelExplainer:
    """Explains ML model predictions using SHAP"""

    def __init__(
        self,
        model,
        X_train: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ):
        """
        Initialize explainer

        Args:
            model: Trained model (XGBoost, RandomForest, etc.)
            X_train: Training data for background distribution
            feature_names: Names of features
        """
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        self.explainer_error = None

    @staticmethod
    def _build_description(
        feature_name: str, current_value: float, shap_value: float
    ) -> str:
        """Build a natural-language explanation for a feature contribution."""
        name = feature_name.lower()
        direction = "bullish" if shap_value >= 0 else "bearish"

        if "rsi" in name:
            if current_value >= 70:
                zone = "overbought zone"
            elif current_value <= 30:
                zone = "oversold zone"
            else:
                zone = "neutral zone"
            return f"RSI at {current_value:.2f} is in {zone}, pushing {direction}."

        if "macd_hist" in name or "macd_histogram" in name:
            trend = "positive" if current_value >= 0 else "negative"
            return f"MACD histogram is {trend} at {current_value:.4f}, supporting {direction} momentum."

        if "bollinger" in name or "bb_" in name:
            return f"Bollinger indicator value {current_value:.4f} contributes a {direction} signal."

        if "volume" in name or "obv" in name:
            return f"Volume-based feature at {current_value:.4f} contributes {direction} pressure."

        return f"{feature_name} at {current_value:.4f} pushes the prediction in a {direction} direction."

    def _format_top_features(
        self,
        feature_names: List[str],
        shap_vals: np.ndarray,
        sample: np.ndarray,
        top_k: int = 10,
    ) -> List[dict]:
        importance_idx = np.argsort(np.abs(shap_vals))[::-1][:top_k]
        top_items: List[dict] = []
        for i in importance_idx:
            shap_val = float(shap_vals[i])
            current_value = float(sample[i])
            top_items.append(
                {
                    "feature_name": feature_names[i],
                    "shap_value": shap_val,
                    "direction": "bullish" if shap_val >= 0 else "bearish",
                    "current_value": current_value,
                    "description": self._build_description(
                        feature_names[i], current_value, shap_val
                    ),
                }
            )
        return top_items

    def create_explainer(self, model_type: str = "xgboost"):
        """
        Create SHAP explainer based on model type

        Args:
            model_type: Type of model ('xgboost', 'random_forest', 'kernel')
        """
        try:
            if model_type == "xgboost":
                self.explainer = shap.TreeExplainer(self.model.model)
            elif model_type == "random_forest":
                self.explainer = shap.TreeExplainer(self.model.model)
            else:
                # Use Kernel Explainer as fallback
                if self.X_train is None:
                    raise ValueError("X_train is required for kernel SHAP explainer.")
                self.explainer = shap.KernelExplainer(
                    self.model.predict, shap.sample(self.X_train, 100)
                )
            self.explainer_error = None
            print(f"✓ {model_type} explainer created")
        except Exception as e:
            self.explainer = None
            self.explainer_error = str(e)
            # Some SHAP/XGBoost version combinations fail to parse booster params.
            # We silently fall back to native XGBoost contribution explanations.
            if model_type == "xgboost":
                print(
                    "Info: SHAP unavailable for current XGBoost build; using native contribution fallback."
                )
            else:
                print(f"Error creating explainer: {str(e)}")

    def _explain_with_xgboost_contribs(
        self, X_test: np.ndarray, index: int = 0
    ) -> dict:
        """Fallback explanation using XGBoost native feature contributions."""
        if index < 0:
            index = X_test.shape[0] + index

        if index < 0 or index >= X_test.shape[0]:
            raise IndexError("index out of range for X_test")

        sample = X_test[index]
        booster = self.model.model.get_booster()
        dmatrix = xgb.DMatrix(X_test)
        contribs = booster.predict(dmatrix, pred_contribs=True)

        # Last column is the expected value (bias term).
        shap_vals = contribs[index][:-1]
        base_value = float(contribs[index][-1])
        prediction = float(self.model.predict(X_test[index : index + 1])[0])

        feature_names = (
            self.feature_names
            if self.feature_names
            else [f"Feature_{i}" for i in range(len(sample))]
        )
        top_features = self._format_top_features(feature_names, shap_vals, sample)

        legacy_map = {
            item["feature_name"]: {
                "shap_value": item["shap_value"],
                "feature_value": item["current_value"],
                "impact": (
                    "Increases" if item["direction"] == "bullish" else "Decreases"
                ),
            }
            for item in top_features
        }

        return {
            "prediction": prediction,
            "base_value": base_value,
            "shap_values": shap_vals,
            "feature_values": sample,
            "feature_names": feature_names,
            "top_features": top_features,
            "top_features_map": legacy_map,
        }

    def explain_prediction(self, X_test: np.ndarray, index: int = 0) -> dict:
        """
        Explain a single prediction

        Args:
            X_test: Test samples
            index: Index of sample to explain

        Returns:
            Dictionary with explanation details
        """
        if self.explainer is None:
            # SHAP can fail for some XGBoost/SHAP version combinations (e.g., base_score parsing).
            # Use native XGBoost contribution values to keep explainability available.
            if hasattr(self.model, "model") and hasattr(
                self.model.model, "get_booster"
            ):
                return self._explain_with_xgboost_contribs(X_test, index)
            raise RuntimeError(
                "SHAP explainer is unavailable. Call create_explainer() and handle failures."
            )

        # Get SHAP values
        if self.explainer is None:
            raise RuntimeError("SHAP explainer is unavailable.")
        explainer = self.explainer
        shap_values = explainer.shap_values(X_test)

        # Handle negative indices properly
        if index < 0:
            index = X_test.shape[0] + index

        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            # For multi-class, take values for positive class
            shap_values = shap_values[1]

        sample = X_test[index]
        shap_vals = shap_values[index]

        top_features = self._format_top_features(
            (
                self.feature_names
                if self.feature_names
                else [f"Feature_{i}" for i in range(len(sample))]
            ),
            shap_vals,
            sample,
        )

        legacy_map = {
            item["feature_name"]: {
                "shap_value": item["shap_value"],
                "feature_value": item["current_value"],
                "impact": (
                    "Increases" if item["direction"] == "bullish" else "Decreases"
                ),
            }
            for item in top_features
        }

        explanation = {
            "prediction": self.model.predict(X_test[index : index + 1])[0],
            "base_value": self.explainer.expected_value,
            "shap_values": shap_vals,
            "feature_values": sample,
            "feature_names": (
                self.feature_names
                if self.feature_names
                else [f"Feature_{i}" for i in range(len(sample))]
            ),
            "top_features": top_features,
            "top_features_map": legacy_map,
        }

        return explanation

    def get_feature_importance_shap(self, X_test: np.ndarray, top_k: int = 10) -> dict:
        """
        Get mean absolute SHAP values (global importance)

        Args:
            X_test: Test data
            top_k: Top k features to return

        Returns:
            Dict of feature importances
        """
        if self.explainer is None:
            self.create_explainer()
        if self.explainer is None:
            raise RuntimeError("SHAP explainer is unavailable.")

        explainer = self.explainer
        shap_values = explainer.shap_values(X_test)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # Mean absolute SHAP values
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

        feature_names = (
            self.feature_names
            if self.feature_names
            else [f"Feature_{i}" for i in range(len(mean_abs_shap))]
        )

        importance_dict = dict(zip(feature_names, mean_abs_shap))
        importance_dict = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        )

        return dict(list(importance_dict.items())[:top_k])

    def explain_prediction_text(
        self, X_test: np.ndarray, index: int = 0, threshold: float = 0.5
    ) -> str:
        """
        Generate human-readable explanation

        Args:
            X_test: Test samples
            index: Sample index
            threshold: Confidence threshold for main drivers

        Returns:
            Text explanation
        """
        explanation = self.explain_prediction(X_test, index)

        pred = explanation["prediction"]
        decision = "BULLISH" if pred > threshold else "BEARISH"
        confidence = pred if pred > 0.5 else 1 - pred

        text = f"""
=== STOCK PREDICTION EXPLANATION ===

Decision: {decision}
Confidence: {confidence:.1%}

KEY DRIVERS:
"""

        if isinstance(explanation["top_features"], list):
            for i, item in enumerate(explanation["top_features"][:10], 1):
                text += f"\n{i}. {item['feature_name']} ({item['direction']}, |SHAP|={abs(item['shap_value']):.4f})"
                text += f"\n   Current value: {item['current_value']:.4f}"
        else:
            for i, (feature, details) in enumerate(
                explanation["top_features"].items(), 1
            ):
                impact = details["impact"]
                shap_val = abs(details["shap_value"])
                feat_val = details["feature_value"]

                text += f"\n{i}. {feature} ({impact} prediction by {shap_val:.4f})"
                text += f"\n   Current value: {feat_val:.4f}"

        text += (
            "\n\nThis explanation shows which factors influenced the prediction most."
        )

        return text


class RuleBasedExplainer:
    """Simple rule-based explanation for interpretability"""

    def __init__(self):
        self.rules = {}

    def add_rule(self, indicator: str, rule_description: str):
        """Add explanation rule for an indicator"""
        self.rules[indicator] = rule_description

    def explain_technical(
        self, rsi: float, macd_hist: float, bb_position: float, trend: str
    ) -> str:
        """
        Explain prediction based on technical indicators

        Args:
            rsi: Relative Strength Index value
            macd_hist: MACD histogram value
            bb_position: Position within Bollinger Bands
            trend: 'UP', 'DOWN', 'SIDEWAYS'

        Returns:
            Text explanation
        """
        explanation = "=== TECHNICAL ANALYSIS ===\n\n"

        # RSI interpretation
        if rsi > 70:
            explanation += f"🔴 RSI ({rsi:.1f}): OVERBOUGHT - expect pullback\n"
        elif rsi < 30:
            explanation += f"🟢 RSI ({rsi:.1f}): OVERSOLD - expect recovery\n"
        else:
            explanation += f"🟡 RSI ({rsi:.1f}): NEUTRAL\n"

        # MACD interpretation
        if macd_hist > 0:
            explanation += f"🟢 MACD: BULLISH crossover\n"
        else:
            explanation += f"🔴 MACD: BEARISH crossover\n"

        # Bollinger Bands
        if bb_position > 0.8:
            explanation += f"🔴 BB Position ({bb_position:.1%}): Near upper band\n"
        elif bb_position < 0.2:
            explanation += f"🟢 BB Position ({bb_position:.1%}): Near lower band\n"
        else:
            explanation += f"🟡 BB Position ({bb_position:.1%}): In middle\n"

        # Trend
        explanation += f"\nMarket Trend: {trend}\n"

        return explanation


def create_summary_explanation(
    prediction: float, top_features: dict, technical_factors: dict
) -> str:
    """
    Create comprehensive explanation combining multiple factors

    Args:
        prediction: Model prediction probability (0-1)
        top_features: Dict of top SHAP values
        technical_factors: Dict of technical indicators

    Returns:
        Comprehensive explanation text
    """

    summary = f"""
╔════════════════════════════════════════════════════════════╗
║           STOCK PREDICTION ANALYSIS REPORT                  ║
╚════════════════════════════════════════════════════════════╝

PREDICTION: {'📈 BULLISH' if prediction > 0.5 else '📉 BEARISH'}
Confidence: {max(prediction, 1-prediction):.1%}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KEY FACTORS (Machine Learning):
"""

    for i, (feature, shap_val) in enumerate(list(top_features.items())[:5], 1):
        if isinstance(shap_val, dict):
            shap_value = shap_val.get("shap_value", 0)
        else:
            shap_value = shap_val
        direction = "↑" if shap_value > 0 else "↓"
        summary += f"{i}. {feature:20} {direction} {abs(shap_value):.4f}\n"

    summary += "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    summary += "\nTECHNICAL INDICATORS:\n"

    for indicator, value in technical_factors.items():
        summary += f"• {indicator:20} {str(value):>15}\n"

    summary += (
        "\n✓ Recommendation: Always use this analysis alongside your own research.\n"
    )

    return summary


def main():
    """Test explainability module"""
    from models import XGBoostPredictor
    from feature_engineering import FeatureEngineer
    from data_ingestion import DataIngestion
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    import numpy as np

    print("Loading and preparing data...")
    ingestion = DataIngestion(lookback_days=365)
    raw_data = ingestion.process_stock_data("RELIANCE.NS")
    if raw_data is None or raw_data.empty:
        print("No usable data returned from ingestion.")
        return

    engineer = FeatureEngineer()
    data = engineer.create_features(raw_data)
    if data is None or data.empty:
        print("Feature generation failed.")
        return

    exclude_cols = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "adj close",
        "target_3d",
        "future_return_3d",
        "target_regression",
        "sector",
    ]
    feature_cols = [col for col in data.columns if col not in exclude_cols]

    X = data[feature_cols].values
    y = data["target_3d"].values

    valid_idx = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X = X[valid_idx]
    y = y[valid_idx]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("Training model...")
    model = XGBoostPredictor()
    model.train(X_train, y_train, X_test, y_test, feature_names=feature_cols)

    print("\nCreating explainer...")
    explainer = ModelExplainer(model, X_train=X_train, feature_names=feature_cols)
    explainer.create_explainer("xgboost")

    print("\nExplaining prediction...")
    _ = explainer.explain_prediction(X_test, index=0)
    print(explainer.explain_prediction_text(X_test, index=0))

    print("\nTop global features (SHAP):")
    importance = explainer.get_feature_importance_shap(X_test)
    for feat, imp in list(importance.items())[:5]:
        print(f"  {feat}: {imp:.6f}")

    print("\n✓ Explainability test complete!")


if __name__ == "__main__":
    main()
