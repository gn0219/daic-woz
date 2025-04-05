from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from collections import defaultdict
import pandas as pd

from collections import Counter
from sklearn.dummy import DummyClassifier

import random
import numpy as np

def set_seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)


class MLPipeline:
    def __init__(self, df, feature_cols, label='PHQ8_Binary', split_col='Split', eval_split='test', seed=42):
        self.df = df
        self.split_col = split_col
        self.eval_split = eval_split
        self.feature_cols = feature_cols
        self.label = label
        self.seed = seed

    def build(self):
        models = [
            ("SVM", SVC(kernel='linear', probability=True)),
            ("KNN", KNeighborsClassifier(n_neighbors=5)),
            ("RandomForest", RandomForestClassifier(n_estimators=100, random_state=self.seed)),
            ("LogisticRegression", LogisticRegression(max_iter=1000, random_state=self.seed)),
            ("GradientBoosting", GradientBoostingClassifier(n_estimators=100, random_state=self.seed)),
            ("XGBoost", XGBClassifier(eval_metric='logloss', random_state=self.seed))
        ]
        return models

    def evaluate_models(self, use_scaler=True, report=False,
                        use_feature_selection=False, selection_method='kbest', 
                        k_features='all', model_selector=None,
                        show_selected_features=False,
                        baseline=False):
        results = []
        reports = defaultdict(dict)

        X_train = self.df.loc[self.df[self.split_col] == 'train', self.feature_cols]
        y_train = self.df.loc[self.df[self.split_col] == 'train', self.label]
        X_eval = self.df.loc[self.df[self.split_col] == self.eval_split, self.feature_cols]
        y_eval = self.df.loc[self.df[self.split_col] == self.eval_split, self.label]

        selected_features = self.feature_cols

        # --- Feature Selection ---
        if use_feature_selection:
            if selection_method == 'kbest':
                selector = SelectKBest(score_func=f_classif, k=k_features)
            elif selection_method == 'model':
                # model_selector: must be a fitted or unfitted model with `coef_` or `feature_importances_`
                if model_selector is None:
                    model_selector = RandomForestClassifier(n_estimators=100, random_state=self.seed)
                selector = SelectFromModel(model_selector)
            else:
                raise ValueError(f"Unknown selection_method: {selection_method}")

            X_train = selector.fit_transform(X_train, y_train)
            X_eval = selector.transform(X_eval)
            selected_mask = selector.get_support(indices=True)
            selected_features = [f for f, s in zip(self.feature_cols, selected_mask) if s]

            if show_selected_features:
                print(f"[Feature Selection: {selection_method}] Selected {len(selected_features)} features:")
                print(selected_features)

        # --- Scaling ---
        if use_scaler:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_eval = scaler.transform(X_eval)

        # --- Majority Class Baseline ---
        if baseline:
            dummy = DummyClassifier(strategy="most_frequent")
            dummy.fit(X_train, y_train)
            y_pred_dummy = dummy.predict(X_eval)

            try:
                y_proba_dummy = dummy.predict_proba(X_eval)[:, 1]
                auroc_dummy = roc_auc_score(y_eval, y_proba_dummy)
            except:
                auroc_dummy = None

            results.append({
                'Model': 'Baseline (Majority Voting)',
                'Accuracy': accuracy_score(y_eval, y_pred_dummy),
                'F1(macro)': f1_score(y_eval, y_pred_dummy, average='macro'),
                'AUROC': auroc_dummy
            })

            if report:
                reports['Baseline (Majority Voting)']['report'] = classification_report(y_eval, y_pred_dummy, output_dict=True)
                reports['Baseline (Majority Voting)']['confusion_matrix'] = confusion_matrix(y_eval, y_pred_dummy)

        # --- Model Training & Evaluation ---
        for model_name, model in self.build():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_eval)

            try:
                y_proba = model.predict_proba(X_eval)[:, 1]
                auroc = roc_auc_score(y_eval, y_proba)
            except:
                auroc = None

            results.append({
                'Model': model_name,
                'Accuracy': accuracy_score(y_eval, y_pred),
                'F1(macro)': f1_score(y_eval, y_pred, average='macro'),
                'AUROC': auroc
            })

            if report:
                reports[model_name]['report'] = classification_report(y_eval, y_pred, output_dict=True)
                reports[model_name]['confusion_matrix'] = confusion_matrix(y_eval, y_pred)

        return (pd.DataFrame(results), reports) if report else pd.DataFrame(results)
