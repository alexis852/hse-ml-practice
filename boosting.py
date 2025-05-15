from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor


sns.set(style='darkgrid')


def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


class Boosting:
    def __init__(
            self,
            base_model_params: dict = None,
            n_estimators: int = 10,
            learning_rate: float = 0.1,
            subsample: float = 0.3,
            early_stopping_rounds: int = None,
            plot: bool = False,
    ):
        self.base_model_class = DecisionTreeRegressor
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators

        self.models: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate
        self.subsample: float = subsample

        self.early_stopping_rounds: int = early_stopping_rounds
        if early_stopping_rounds is not None:
            self.validation_loss = np.full(self.early_stopping_rounds, np.inf)

        self.plot: bool = plot

        self.history = defaultdict(list)

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)
        # self.loss_derivative2 = lambda y, z: y ** 2 * self.sigmoid(-y * z) * (1 - self.sigmoid(-y * z))

    def fit_new_base_model(self, x, y, predictions):
        boot_indexes = np.random.randint(0, x.shape[0], int(self.subsample * x.shape[0]))
        x_boot, y_boot, pred_boot = x[boot_indexes], y[boot_indexes], predictions[boot_indexes]
        new_base_model = self.base_model_class(**self.base_model_params).fit(x_boot, -self.loss_derivative(y_boot, pred_boot))
        self.gammas.append(self.find_optimal_gamma(y, predictions, new_base_model.predict(x)) * self.learning_rate)
        self.models.append(new_base_model)

    def fit(self, x_train, y_train, x_valid, y_valid):
        """
        :param x_train: features array (train set)
        :param y_train: targets array (train set)
        :param x_valid: features array (validation set)
        :param y_valid: targets array (validation set)
        """
        train_predictions = np.zeros(y_train.shape[0])
        valid_predictions = np.zeros(y_valid.shape[0])

        self.history['train'].append(self.score(x_train, y_train))
        self.history['valid'].append(self.score(x_valid, y_valid))

        for i in range(self.n_estimators):
            self.fit_new_base_model(x_train, y_train, train_predictions)

            train_predictions += self.gammas[-1] * self.models[-1].predict(x_train)
            valid_predictions += self.gammas[-1] * self.models[-1].predict(x_valid)

            self.history['train'].append(self.score(x_train, y_train))
            self.history['valid'].append(self.score(x_valid, y_valid))

            if self.early_stopping_rounds is not None:
                valid_loss = self.loss_fn(y_valid, valid_predictions)
                if np.all(valid_loss >= self.validation_loss):
                    break
                self.validation_loss[i % self.early_stopping_rounds] = valid_loss

        if self.plot:
            plt.plot(np.arange(len(self.history['train'])), self.history['train'], label='train')
            plt.plot(np.arange(len(self.history['valid'])), self.history['valid'], label='valid')
            plt.xlabel('n_estimators')
            plt.ylabel('AUC-ROC')
            plt.legend()
            plt.show()

    def predict_proba(self, x):
        y_pred = np.zeros(x.shape[0])
        for gamma, model in zip(self.gammas, self.models):
            y_pred += gamma * model.predict(x)
        y_prob = self.sigmoid(y_pred)
        return np.array([1 - y_prob, y_prob]).T

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]

        return gammas[np.argmin(losses)]

    def score(self, x, y):
        return score(self, x, y)

    @property
    def feature_importances_(self):
        mean_importances = np.mean([model.feature_importances_ for model in self.models], axis=0)
        return mean_importances / mean_importances.sum()
    
    @property
    def _estimator_type(self):
        return 'classifier'

    def get_params(self, deep=True):
        return {'n_estimators': self.n_estimators,
                'learning_rate': self.learning_rate,
                'subsample': self.subsample}