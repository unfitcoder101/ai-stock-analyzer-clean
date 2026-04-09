import matplotlib.pyplot as plt
import seaborn as sns

# 🔥 Replace with YOUR API output values
dt_cm = [[120, 30],
         [40, 110]]

rf_cm = [[120, 30],
         [40, 110]]

lr_cm = [[120, 30],
         [40, 110]]

labels = ["DOWN", "UP"]

# DT
plt.figure()
sns.heatmap(dt_cm, annot=True, fmt="d", cmap="coolwarm",
            xticklabels=labels, yticklabels=labels)
plt.title("Decision Tree Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("dt_heatmap.png")

# RF
plt.figure()
sns.heatmap(rf_cm, annot=True, fmt="d", cmap="coolwarm",
            xticklabels=labels, yticklabels=labels)
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("rf_heatmap.png")

# LR
plt.figure()
sns.heatmap(lr_cm, annot=True, fmt="d", cmap="coolwarm",
            xticklabels=labels, yticklabels=labels)
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("lr_heatmap.png")

print("✅ Heatmaps saved")
