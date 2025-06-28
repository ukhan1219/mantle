from exporters.coreml.features import FeaturesManager

llama_features = list(FeaturesManager.get_supported_features_for_model_type("llama").keys())
print("Supported features for Llama:")
print(llama_features)
