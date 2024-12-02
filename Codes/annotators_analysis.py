import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import fleiss_kappa

# Step 1: Load the data from Excel
df = pd.read_excel("Annotation_data.xlsx")

# Step 2: Calculate Cohen's Kappa for pairwise annotators
kappa_1_2 = cohen_kappa_score(df["annotator 1"], df["annotator 2"])
kappa_1_3 = cohen_kappa_score(df["annotator 1"], df["annotator 3"])
kappa_2_3 = cohen_kappa_score(df["annotator 2"], df["annotator 3"])

# Print Cohen's Kappa
print("Cohen's Kappa:")
print(f"Annotator 1 vs Annotator 2: {kappa_1_2:.2f}")
print(f"Annotator 1 vs Annotator 3: {kappa_1_3:.2f}")
print(f"Annotator 2 vs Annotator 3: {kappa_2_3:.2f}")

# Step 3: Calculate Fleiss' Kappa for all annotators
# Prepare data for Fleiss' Kappa calculation
annotation_counts = df[["annotator 1", "annotator 2", "annotator 3"]].apply(
    lambda row: pd.Series([sum(row == 0), sum(row == 1)]), axis=1
)
annotation_counts.columns = [0, 1]  # Categories: 0 and 1

# Calculate Fleiss' Kappa
fleiss_kappa_score = fleiss_kappa(annotation_counts.values, method='fleiss')
print("\nFleiss' Kappa:")
print(f"Overall agreement (3 annotators): {fleiss_kappa_score:.2f}")

# Step 4: Analyze annotator disagreement
df["disagreement"] = df[["annotator 1", "annotator 2", "annotator 3"]].nunique(axis=1) > 1
disagreement_words = df[df["disagreement"]]["word"].tolist()

print("\nWords with disagreement among annotators:")
print(disagreement_words)

# Step 5: Statistics on annotations
# Calculate the frequency of each label per annotator
label_counts = {}
for annotator in ["annotator 1", "annotator 2", "annotator 3"]:
    label_counts[annotator] = df[annotator].value_counts()

# Step 6: Visualization
# 1. Bar plot for label counts
plt.figure(figsize=(10, 6))
for i, annotator in enumerate(label_counts, start=1):
    plt.bar([f"{annotator} (0)", f"{annotator} (1)"], label_counts[annotator], alpha=0.7, label=annotator)
plt.title("Label Counts for Each Annotator")
plt.ylabel("Count")
plt.xlabel("Labels")
plt.legend(loc="upper right")
plt.show()

# 2. Cohen's Kappa Pairwise Scores
kappa_scores = {
    "Annotator 1 vs 2": kappa_1_2,
    "Annotator 1 vs 3": kappa_1_3,
    "Annotator 2 vs 3": kappa_2_3,
}
plt.figure(figsize=(8, 5))
plt.bar(kappa_scores.keys(), kappa_scores.values(), color='skyblue')
plt.ylim(0, 1)
plt.title("Cohen's Kappa Scores (Pairwise Annotators)")
plt.ylabel("Kappa Score")
plt.xlabel("Annotator Pairs")
plt.axhline(y=0.6, color="red", linestyle="--", label="Threshold")
plt.legend()
plt.show()

# 3. Words with Disagreement (Bar Chart)
disagreement_count = len(disagreement_words)
total_words = len(df)
agreement_count = total_words - disagreement_count

plt.figure(figsize=(6, 6))
plt.pie(
    [agreement_count, disagreement_count],
    labels=["Agreement", "Disagreement"],
    autopct="%1.1f%%",
    startangle=90,
    colors=["lightgreen", "salmon"],
)
plt.title("Agreement vs Disagreement Among Annotators")
plt.show()



