from .model_runtime import compare_all_model_predictions as runtime_compare_all_model_predictions
from .model_runtime import compare_model_predictions as runtime_compare_model_predictions
from .model_runtime import predict_message as runtime_predict_message


def predict_message(message: str, output_language: str = "English", model_kind: str = "best"):
    return runtime_predict_message(message, output_language=output_language, model_kind=model_kind)


def compare_model_predictions(
    message: str,
    output_language: str = "English",
    primary_model_kind: str | None = "best",
    secondary_model_kind: str | None = None,
):
    return runtime_compare_model_predictions(
        message,
        output_language=output_language,
        primary_model_kind=primary_model_kind,
        secondary_model_kind=secondary_model_kind,
    )


def compare_all_model_predictions(message: str, output_language: str = "English"):
    return runtime_compare_all_model_predictions(message, output_language=output_language)
