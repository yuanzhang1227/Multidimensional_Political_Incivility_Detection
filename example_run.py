import pandas as pd
from load_clfs_IMP_PHAVPR_HSST_THREAT import predict_for_lists_of_text

# Load example input data
df = pd.read_csv("example_input.csv")
texts = df["text"].tolist()

# Run prediction
results = predict_for_lists_of_text(texts)

# Combine results into a DataFrame
output_df = pd.DataFrame({
    "post_id": df["post_id"],
    "text": texts,
    "label_IMP": results["labels_IMP"],
    "prob_IMP_1": results["probas_class_1_IMP"],
    "label_PHAVPR": results["labels_PHAVPR"],
    "prob_PHAVPR_1": results["probas_class_1_PHAVPR"],
    "label_HSST": results["labels_HSST"],
    "prob_HSST_1": results["probas_class_1_HSST"],
    "label_THREAT": results["labels_THREAT"],
    "prob_THREAT_1": results["probas_class_1_THREAT"]
})

# Save to CSV
output_df.to_csv("Prediction_Multidimensional_Incivility.csv", index=False)
print("Predictions saved to 'Prediction_Multidimensional_Incivility.csv'")
