import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.initializers import LecunUniform
from tensorflow.keras.activations import relu, tanh, linear
from tensorflow.keras.losses import mse
from tensorflow.keras.models import load_model
from joblib import dump, load


class Model:
    def __init__(self, env, state_samples, train_mode):
        self.env = env
        self.train_mode = train_mode
        if self.train_mode:
            self.scaler = StandardScaler()
            self.scaler.fit(state_samples)
            self.models = [self.create_nn_model() for _ in range(self.env.action_space.n)]

    def create_nn_model(self):
        model = Sequential()
        model.add(Dense(128, kernel_initializer=LecunUniform(), activation=relu, input_shape=(8,)))
        model.add(Dense(256, kernel_initializer=LecunUniform(), activation=tanh))
        model.add(Dense(1, kernel_initializer=LecunUniform(), activation=linear))
        model.compile(loss=mse, optimizer=Adamax())
        return model

    def predict(self, s):
        X = self.scaler.transform(np.atleast_2d(s))
        return np.array([m.predict(np.array(X), verbose=0)[0] for m in self.models])

    def update(self, s, a, G):
        X = self.scaler.transform(np.atleast_2d(s))
        self.models[a].fit(np.array(X), np.array([G]), epochs=1, verbose=0)

    def sample_action(self, s, eps):
        return self.env.action_space.sample() if np.random.random() < eps and self.train_mode else np.argmax(self.predict(s))

    def save(self):
        dump(self.scaler, '../saves_old/scaler.joblib')
        [model.save(f"../saves_old/model{i}.h5") for i, model in enumerate(self.models)]

    def load(self):
        self.scaler = load('../saves_old/scaler.joblib')
        self.models = [load_model(f"../saves_old/model{i}.h5") for i in range(self.env.action_space.n)]
