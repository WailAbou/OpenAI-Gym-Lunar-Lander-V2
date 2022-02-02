from typing import Any, List
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow import config as tf_config
import tensorflow.compat.v1 as tf
import warnings
import numpy as np


warnings.filterwarnings("ignore")
tf.disable_v2_behavior()
gpu_amount = len(tf_config.experimental.list_physical_devices('GPU'))
print(f'GPU acceleration enabled = {gpu_amount > 0}')


class FunctionApproximator:
    def __init__(self, learning_rate: float = 0.001) -> None:
        self.model = Sequential()
        self.model.add(Dense(1, input_dim=8))
        self.model.add(Dense(32))
        self.model.add(Dense(64))
        self.model.add(Dense(4))
        self.model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
        
    def q_values(self, states: List[Any]) -> Any:
        return self.model.predict(np.array([states]))[0]

    def save(self, file_name: str) -> None:
        self.model.save(f'saves_new/{file_name}.h5')

    def load(self, file_name: str) -> None:
        self.model = load_model(f'saves_new/{file_name}.h5')

    def train(self, data, targets) -> None:
        self.model.fit(np.array(data), np.array(targets), verbose=0)

    def set_weights(self, new_weights) -> None:
        [layer.set_weights(new_weights[i]) for i, layer in enumerate(self.model.layers)]

    def get_weights(self) -> List[List[float]]:
        return np.array([layer.get_weights() for layer in self.model.layers])