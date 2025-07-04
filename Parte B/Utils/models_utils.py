"""
Este módulo define la clase Models, que centraliza la creación de modelos de machine learning,
vectorizadores y técnicas de sobremuestreo para tareas de clasificación de texto.
Incluye modelos clásicos como regresión logística, random forest, SVM, árboles de decisión,
k-NN, Naive Bayes y XGBoost, así como utilidades para vectorización TF-IDF y SMOTE.
"""
from skorch import NeuralNetClassifier  # type: ignore
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore
from sklearn.linear_model import LogisticRegression # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.svm import SVC # type: ignore
from sklearn.tree import DecisionTreeClassifier # type: ignore
from sklearn.naive_bayes import MultinomialNB # type: ignore
from sklearn.neighbors import KNeighborsClassifier # type: ignore
from sklearn.ensemble import ExtraTreesClassifier # type: ignore
import lightgbm as lgb # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from imblearn.over_sampling import SMOTE # type: ignore
import xgboost as xgb

class SimpleFFNN(nn.Module):
    def __init__(self, input_dim=80):
        super(SimpleFFNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class Models:
    """
    Clase utilitaria para instanciar modelos de machine learning, vectorizadores y técnicas de sobremuestreo.
    """

    @staticmethod
    def _logistic_regression():
        """
        Crea una instancia de regresión logística con parámetros predefinidos.
        """
        return LogisticRegression(
            solver='liblinear', 
            dual=False, 
            fit_intercept=True, 
            intercept_scaling=1, 
            max_iter=1000,  
            verbose=0, 
            warm_start=False, 
            penalty="l2", 
            random_state=42
        )

    @staticmethod
    def _random_forest():
        """
        Crea una instancia de Random Forest con parámetros predefinidos y balanceo de clases.
        """
        return RandomForestClassifier(
            max_depth=30,
            max_features='sqrt',
            min_samples_leaf=5,
            min_samples_split=2,
            n_estimators=500,
            class_weight='balanced'
        )

    @staticmethod
    def _svm():
        """
        Crea una instancia de SVM (Support Vector Machine) con kernel RBF y balanceo de clases.
        """
        return SVC(
            probability=True,
            C=1,
            gamma='scale',
            kernel='rbf',
            class_weight='balanced'
        )

    @staticmethod
    def _decision_tree():
        """
        Crea una instancia de árbol de decisión con parámetros predefinidos y balanceo de clases.
        """
        return DecisionTreeClassifier(
            criterion='entropy', 
            max_depth=5, 
            max_leaf_nodes=5,
            min_samples_leaf=10, 
            min_samples_split=2,
            class_weight='balanced'
        )
    
    @staticmethod
    def _k_nearest_neighbors():
        """
        Crea una instancia de K-Nearest Neighbors con parámetros predefinidos.
        """
        return KNeighborsClassifier(
            n_neighbors=190, 
            weights="uniform", 
            metric="cosine"
        )
    
    @staticmethod
    def _naive_bayes():
        """
        Crea una instancia de Naive Bayes multinomial.
        """
        return MultinomialNB()
    
    @staticmethod
    def _xgboost(**kwargs):
        """
        Crea una instancia de XGBoost para clasificación binaria con parámetros por defecto,
        permitiendo sobreescribirlos mediante kwargs.
        """
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
    def lightgbm(**kwargs):
        default_params = {
            'objective': 'binary',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'n_estimators': 100,
            'class_weight': 'balanced',
            'random_state': 42
        }
        default_params.update(kwargs)
        return lgb.LGBMClassifier(**default_params)

    @staticmethod
    def _deep_ffnn():
        """
        Crea una red neuronal profunda simple usando PyTorch + skorch.
        """
        return NeuralNetClassifier(
            module=SimpleFFNN,
            module__input_dim=80,
            max_epochs=20,
            lr=0.001,
            optimizer=torch.optim.Adam,
            iterator_train__shuffle=True,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            criterion=torch.nn.BCEWithLogitsLoss
        )


    @staticmethod
    def get_models():
        """
        Devuelve un diccionario con instancias de todos los modelos implementados.
        """
        return {
            # "Logistic Regression": Models._logistic_regression(),
            # "Random Forest": Models._random_forest(),
            # "K-Nearest Neighbors": Models._k_nearest_neighbors(),
            # "SVM": Models._svm(),   
            # "Decision Tree": Models._decision_tree(),
            # "Naive Bayes": Models._naive_bayes(),
            # "XGBoost": Models._xgboost(),
            # "Extra Trees": Models.extra_trees(),
            # "LightGBM": Models.lightgbm(),
            "Deep FFNN": Models._deep_ffnn()
        }
    
    @staticmethod
    def get_vectorizer():
        """
        Devuelve una instancia de TfidfVectorizer configurada para procesamiento de texto.
        """
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
        """
        Devuelve una instancia de SMOTE para sobremuestreo de clases minoritarias.
        """
        return SMOTE(random_state=42)
