import shap


def create_explanation(sample_id):
  explainer = shap.TreeExplainer(model)
  instance = X_test.iloc[sample_id:sample_id+1]
  class_idx = y_test.iloc[sample_id:sample_id+1].values[0]
  print("class_idx", class_idx)
  shap_values = explainer(instance)
  shap.initjs()
  shap.plots.waterfall(shap_values[0][:, class_idx], max_display=11, show=True)

def get_top_influence_features(k):
  explainer = shap.TreeExplainer(model)
  feature_names = X_test.columns
  keys = [
      "Social Engagement",
      "Linguistic & Readability",
      "Communication Style",
      "Emotion & Sentiment",
      "Activity day",
      "Activity time",
      "Client Utilization"
  ]
  feature_dict = {key: 0 for key in keys}
  for i in range(len(X_test):
    instance = X_test.iloc[i:i+1]
    class_idx = y_test.iloc[i:i+1].values[0]
    shap_values = explainer(instance)
    abs_shap_values = np.abs(shap_values.values[0][:,class_idx])
    top_indices = np.argsort(abs_shap_values)[-k:]
    for index in top_indices:
      feature_name = feature_names[index]
      category = feature_to_category.get(feature_name, None)
      feature_dict[category] += 1
  total_top_features = sum(feature_dict.values())
  for key in keys:
          percentage = (feature_dict[key] / total_top_features) * 100
          print(f"{key}: {percentage:.2f}%")
  
