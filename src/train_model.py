# src/train_model.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import xgboost as xgb

TRAIN_FEATURES = "data/features_merged.csv"
OUT_MODEL = "models/xgb_final.json"
OUT_SUB = "submission/submission.csv"

df = pd.read_csv(TRAIN_FEATURES)
# drop rows without label
df = df.dropna(subset=["label"])
# features to use (exclude filename, transcript, label)
exclude = {"filename","transcript","label"}
features = [c for c in df.columns if c not in exclude]

X = df[features].fillna(0).values
y = df["label"].values

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(df))
fold = 0
rmses = []
pearsons = []

for train_idx, val_idx in kf.split(X):
    fold += 1
    Xtr, Xval = X[train_idx], X[val_idx]
    ytr, yval = y[train_idx], y[val_idx]
    dtrain = xgb.DMatrix(Xtr, label=ytr)
    dval = xgb.DMatrix(Xval, label=yval)
    params = {"objective":"reg:squarederror", "max_depth":6, "eta":0.1, "seed":42}
    bst = xgb.train(params, dtrain, num_boost_round=300, evals=[(dval,"val")], early_stopping_rounds=20, verbose_eval=False)
    pred = bst.predict(dval)
    oof[val_idx] = pred
    rmse = mean_squared_error(yval, pred, squared=False)
    rmses.append(rmse)
    # pearson
    pearson = np.corrcoef(yval, pred)[0,1]
    pearsons.append(pearson)
    print(f"Fold {fold}: RMSE={rmse:.4f}, Pearson={pearson:.4f}")

print("CV RMSE mean:", np.mean(rmses))
print("CV Pearson mean:", np.mean(pearsons))

# TRAIN RMSE required: compute on full training set using OOF
train_rmse = mean_squared_error(y, oof, squared=False)
print("TRAIN RMSE (oof):", train_rmse)

# Train final on full data
dall = xgb.DMatrix(X, label=y)
final = xgb.train(params, dall, num_boost_round=bst.best_iteration)
os.makedirs(os.path.dirname(OUT_MODEL), exist_ok=True)
final.save_model(OUT_MODEL)
print("Saved model to", OUT_MODEL)

# Create dummy submission if test features exist
TEST_FEATURES = "data/features_test.csv"  # optional if you build test features
if os.path.exists(TEST_FEATURES):
    test_df = pd.read_csv(TEST_FEATURES)
    test_X = test_df[features].fillna(0).values
    preds = final.predict(xgb.DMatrix(test_X))
    sub = pd.DataFrame({"filename": test_df["filename"], "label": preds})
    os.makedirs(os.path.dirname(OUT_SUB), exist_ok=True)
    sub.to_csv(OUT_SUB, index=False)
    print("Saved submission:", OUT_SUB)
else:
    print("No test features found; skip submission save.")
