import shap
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load and prepare data
data = load_diabetes(as_frame=True)
df = data.frame.copy()
df['target'] = (df['target'] > df['target'].mean()).astype(int)
X = df.drop('target', axis=1)
y = df['target']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# SHAP explanation
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_train)

# Plot and save
plt.figure()
shap.summary_plot(shap_values, X_train, show=False)
plt.title("SHAP Summary Plot")
plt.savefig("results/shap_summary_plot.png", bbox_inches="tight")
plt.close()
