from sklearn.linear_model import LogisticRegression # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.svm import SVC # type: ignore
from sklearn.tree import DecisionTreeClassifier # type: ignore
from sklearn.naive_bayes import MultinomialNB # type: ignore
from sklearn.neighbors import KNeighborsClassifier # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from sklearn.ensemble import ExtraTreesClassifier # type: ignore
from imblearn.over_sampling import SMOTE # type: ignore
import xgboost as xgb


class Models:
    @staticmethod
    def logistic_regression():
        return LogisticRegression(
            solver='liblinear', 
            dual=False, 
            fit_intercept=True, 
            intercept_scaling=1, 
            max_iter=1000, 
            multi_class='ovr', 
            verbose=0, 
            warm_start=False, 
            penalty="l2", 
            random_state=42
        )

    @staticmethod
    def random_forest():
        return RandomForestClassifier(
            max_depth=30,
            max_features='sqrt',
            min_samples_leaf=5,
            min_samples_split=2,
            n_estimators=500,
            class_weight='balanced'
        )

    @staticmethod
    def svm():
        return SVC(
            probability=True,
            C=1,
            gamma='scale',
            kernel='rbf',
            class_weight='balanced'
        )

    @staticmethod
    def decision_tree():
        return DecisionTreeClassifier(
            criterion='entropy', 
            max_depth=5, 
            max_leaf_nodes=5,
            min_samples_leaf=10, 
            min_samples_split=2,
            class_weight='balanced'
        )
    
    @staticmethod
    def k_nearest_neighbors():
        return KNeighborsClassifier(
            n_neighbors=190, 
            weights="uniform", 
            metric="cosine"
        )
    
    @staticmethod
    def naive_bayes():
        return MultinomialNB()
    
    @staticmethod
    def xgboost(**kwargs):
        default_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'learning_rate': 0.05,
            'max_depth': 6,
            'n_estimators': 100,
            'use_label_encoder': False,
            'random_state': 42
        }
        default_params.update(kwargs)
        return xgb.XGBClassifier(**default_params)

    @staticmethod
    def extra_trees():
        return ExtraTreesClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=False,
            random_state=42
        )

    @staticmethod
    def get_models():
        return {
            "Logistic Regression": Models.logistic_regression(),
            "Random Forest": Models.random_forest(),
            "K-Nearest Neighbors": Models.k_nearest_neighbors(),
            "SVM": Models.svm(),   
            "Decision Tree": Models.decision_tree(),
            "Naive Bayes": Models.naive_bayes(),
            "XGBoost": Models.xgboost(),
            "Extra Trees": Models.extra_trees()
        }
    
    @staticmethod
    def get_vectorizer():
        return TfidfVectorizer(
            ngram_range=(1, 4),
            max_df=0.95,
            min_df=2,
            max_features=10000,
            sublinear_tf=True,
            stop_words='english',
            lowercase=True
        )
    
    @staticmethod
    def get_smote():
        return SMOTE(random_state=42)
    
    