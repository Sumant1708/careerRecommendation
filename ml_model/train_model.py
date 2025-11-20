import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

df = pd.read_csv("Career_Dataset_10000_Realistic.csv")

# Combine Skills + Interests into a single text feature
df["Combined_Text"] = df["Skills"] + ";" + df["Interests"]

X = df[["Age", "Education", "Combined_Text"]]
y = df["Recommended_Career"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

text_features = ["Combined_Text"]
numeric_features = ["Age"]
cat_features = ["Education"]

preprocessor = ColumnTransformer(
    transformers=[
        ("tfidf", TfidfVectorizer(
            max_features=1500,
            token_pattern=r"[^;]+",     # treat each skill/interest as a token
            lowercase=True
        ), "Combined_Text"),
        ("ohe", OneHotEncoder(handle_unknown="ignore"), ["Education"]),
        ("scaler", StandardScaler(), ["Age"])
    ],
    sparse_threshold=0.3
    )

model = RandomForestClassifier(
    n_estimators=250,
    random_state=42,
    n_jobs=-1,
    max_depth=None
)

pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("classifier", model)
])

pipeline.fit(X_train, y_train)


y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {acc*100:.2f}\n")

joblib.dump(pipeline, "career_recommender_model.pkl")
print("✅ Model saved as career_recommender_model.pkl")