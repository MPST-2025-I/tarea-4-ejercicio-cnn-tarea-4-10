import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score
import tensorflow as tf
import os
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Flatten, Input, concatenate, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanAbsoluteError, MeanSquaredError
from sklearn.model_selection import train_test_split


class MelanieSolution:
    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.fcm = None
        self.cm = None
        self.history = None

    def preprocessing(self, df):
        X = df.drop(columns=['price'])
        y = df['price']

        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

        X[numerical_cols] = X[numerical_cols].fillna(X[numerical_cols].median())

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
            ]
        )

        X_processed = preprocessor.fit_transform(X)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42
        )
        return self

    def fully_connected_model(self):
        model = Sequential([
            Input(shape=(self.X_train.shape[1],)),  # Specify input shape with Input layer
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        self.fcm = model
        return self

    def convolutional_model(self):
        X_train_reshaped = self.X_train.reshape(self.X_train.shape[0], self.X_train.shape[1], 1)
        X_test_reshaped = self.X_test.reshape(self.X_test.shape[0], self.X_test.shape[1], 1)

        model = Sequential([
            Input(shape=(X_train_reshaped.shape[1], 1)),  # Specify input shape with Input layer
            Conv1D(filters=64, kernel_size=3, activation='relu'),
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        self.cm = model
        return self

    def evaluate(self, model=None):
        model_to_evaluate = model if model else self.fcm

        history = model_to_evaluate.fit(
            self.X_train, self.y_train,
            epochs=400, batch_size=16,
            validation_data=(self.X_test, self.y_test),
            verbose=0
        )

        y_pred = model_to_evaluate.predict(self.X_test).flatten()
        r2 = r2_score(self.y_test, y_pred)
        print(f'R² Score: {r2:.4f}')

        self.history = history
        return self

    def run_pipeline(self, df):
        self.preprocessing(df)
        self.fully_connected_model()
        self.evaluate()
        self.convolutional_model()
        self.evaluate(model=self.cm)
        return self

class MelanieSolution2:
    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}  # Diccionario para almacenar los modelos
        self.histories = {}  # Diccionario para almacenar los historiales de entrenamiento
        self.results = {}  # Diccionario para almacenar los resultados finales

    def load_data(self, folder_path):
        """Cargar todos los archivos CSV en la carpeta."""
        all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
        df_list = [pd.read_csv(file) for file in all_files]
        df = pd.concat(df_list, ignore_index=True)
        return df

    def preprocessing(self, df, target_column, n_steps_in, n_steps_out):
        """Preprocesar los datos y dividirlos en secuencias."""
        # Convertir SETTLEMENTDATE a datetime
        df['SETTLEMENTDATE'] = pd.to_datetime(df['SETTLEMENTDATE'])

        # Extraer características temporales
        df['HOUR'] = df['SETTLEMENTDATE'].dt.hour
        df['DAYOFWEEK'] = df['SETTLEMENTDATE'].dt.dayofweek
        df['MONTH'] = df['SETTLEMENTDATE'].dt.month

        # Seleccionar características y objetivo
        features = ['HOUR', 'DAYOFWEEK', 'MONTH', 'RRP', 'PERIODTYPE']
        X = df[features]
        y = df[target_column]

        # Codificar PERIODTYPE (si es categórico)
        X = pd.get_dummies(X, columns=['PERIODTYPE'], drop_first=True)

        # Normalizar los datos
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Dividir la serie temporal en secuencias con múltiples pasos futuros
        X_seq, y_seq = self.split_sequence(X_scaled, y.values, n_steps_in, n_steps_out)

        # Dividir en conjuntos de entrenamiento y prueba
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_seq, y_seq, test_size=0.2, random_state=42
        )

        return self

    def split_sequence(self, X, y, n_steps_in, n_steps_out):
        """Dividir una secuencia en muestras con múltiples pasos futuros."""
        X_seq, y_seq = list(), list()
        for i in range(len(X)):
            end_ix = i + n_steps_in
            out_end_ix = end_ix + n_steps_out

            if out_end_ix > len(X):
                break

            seq_x = X[i:end_ix]
            seq_y = y[end_ix:out_end_ix]

            X_seq.append(seq_x)
            y_seq.append(seq_y)

        return np.array(X_seq), np.array(y_seq)

    def build_univariate_cnn(self):
        """Construir un modelo CNN univariado."""
        model = Sequential([
            Input(shape=(self.X_train.shape[1], self.X_train.shape[2])),
            Conv1D(filters=64, kernel_size=3, activation='relu'),
            MaxPooling1D(),
            Flatten(),
            Dense(50, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=[MeanAbsoluteError()])
        self.models['univariate_cnn'] = model
        return self

    def build_multivariate_cnn(self):
        """Construir un modelo CNN multivariado."""
        model = Sequential([
            Input(shape=(self.X_train.shape[1], self.X_train.shape[2])),
            Conv1D(filters=64, kernel_size=3, activation='relu'),
            MaxPooling1D(),
            Flatten(),
            Dense(50, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=[MeanAbsoluteError()])
        self.models['multivariate_cnn'] = model
        return self

    def build_multiple_header_cnn(self):
        """Construir un modelo CNN con múltiples cabeceras."""
        input1 = Input(shape=(self.X_train.shape[1], 1))
        cnn1 = Conv1D(filters=64, kernel_size=3, activation='relu')(input1)
        cnn1 = MaxPooling1D()(cnn1)
        cnn1 = Flatten()(cnn1)

        input2 = Input(shape=(self.X_train.shape[1], 1))
        cnn2 = Conv1D(filters=64, kernel_size=3, activation='relu')(input2)
        cnn2 = MaxPooling1D()(cnn2)
        cnn2 = Flatten()(cnn2)

        merged = concatenate([cnn1, cnn2])
        dense = Dense(50, activation='relu')(merged)
        output = Dense(1)(dense)

        model = Model(inputs=[input1, input2], outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=[MeanAbsoluteError()])
        self.models['multiple_header_cnn'] = model
        return self

    def build_multiple_output_cnn(self):
        """Construir un modelo CNN con múltiples salidas."""
        input_layer = Input(shape=(self.X_train.shape[1], self.X_train.shape[2]))
        cnn = Conv1D(filters=64, kernel_size=3, activation='relu')(input_layer)
        cnn = MaxPooling1D()(cnn)
        cnn = Flatten()(cnn)
        cnn = Dense(50, activation='relu')(cnn)

        output1 = Dense(1)(cnn)
        output2 = Dense(1)(cnn)
        output3 = Dense(1)(cnn)

        model = Model(inputs=input_layer, outputs=[output1, output2, output3])

        # Compilar el modelo con una métrica para cada salida
        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='mse',
                      metrics=[MeanAbsoluteError(), MeanAbsoluteError(), MeanAbsoluteError()])

        self.models['multiple_output_cnn'] = model
        return self

    def build_multiple_steps_cnn(self):
        """Construir un modelo CNN para predicción de múltiples pasos."""
        model = Sequential([
            Input(shape=(self.X_train.shape[1], self.X_train.shape[2])),
            Conv1D(filters=64, kernel_size=3, activation='relu'),
            MaxPooling1D(),
            Flatten(),
            Dense(50, activation='relu'),
            Dense(2)  # Predice 2 pasos futuros
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=[MeanAbsoluteError()])
        self.models['multiple_steps_cnn'] = model
        return self

    def evaluate_models(self, epochs=10, batch_size=32):
        """Entrenar y evaluar todos los modelos."""
        for model_name, model in self.models.items():
            print(f"\nEntrenando el modelo: {model_name}")
            if model_name == 'multiple_header_cnn':
                # Preparar datos para el modelo de múltiples cabeceras
                X1 = self.X_train[:, :, 0].reshape(self.X_train.shape[0], self.X_train.shape[1], 1)
                X2 = self.X_train[:, :, 1].reshape(self.X_train.shape[0], self.X_train.shape[1], 1)
                history = model.fit([X1, X2], self.y_train, epochs=epochs, batch_size=batch_size,
                                    validation_data=(
                                    [self.X_test[:, :, 0].reshape(self.X_test.shape[0], self.X_test.shape[1], 1),
                                     self.X_test[:, :, 1].reshape(self.X_test.shape[0], self.X_test.shape[1], 1)],
                                    self.y_test))

                # Evaluar el modelo de múltiples cabeceras
                X1_test = self.X_test[:, :, 0].reshape(self.X_test.shape[0], self.X_test.shape[1], 1)
                X2_test = self.X_test[:, :, 1].reshape(self.X_test.shape[0], self.X_test.shape[1], 1)
                loss, mae = model.evaluate([X1_test, X2_test], self.y_test, verbose=0)
                self.results[model_name] = {'Loss': loss, 'MAE': mae}

            elif model_name == 'multiple_output_cnn':
                # Preparar datos para el modelo de múltiples salidas
                y1 = self.y_train[:, 0].reshape((self.y_train.shape[0], 1))  # Primera salida
                y2 = self.y_train[:, 1].reshape((self.y_train.shape[0], 1))  # Segunda salida
                y3 = self.y_train[:, 2].reshape((self.y_train.shape[0], 1))  # Tercera salida

                # Entrenar el modelo
                history = model.fit(self.X_train, [y1, y2, y3], epochs=epochs, batch_size=batch_size,
                                    validation_data=(self.X_test, [self.y_test[:, 0].reshape((self.y_test.shape[0], 1)),
                                                                   self.y_test[:, 1].reshape((self.y_test.shape[0], 1)),
                                                                   self.y_test[:, 2].reshape(
                                                                       (self.y_test.shape[0], 1))]))

                # Evaluar el modelo de múltiples salidas
                y_pred = model.predict(self.X_test)
                mse = np.mean((self.y_test - y_pred[0].flatten()) ** 2)
                self.results[model_name] = {'MSE': mse}

            else:
                # Entrenar otros modelos
                history = model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size,
                                    validation_data=(self.X_test, self.y_test))

                # Evaluar otros modelos
                loss, mae = model.evaluate(self.X_test, self.y_test, verbose=0)
                self.results[model_name] = {'Loss': loss, 'MAE': mae}

            self.histories[model_name] = history
        return self

    def print_summary(self):
        """Imprimir un resumen de los resultados de todos los modelos."""
        print("\nResumen de Resultados:")
        for model_name, result in self.results.items():
            print(f"\nModelo: {model_name}")
            for metric, value in result.items():
                print(f"{metric}: {value:.4f}")

    def preprocessing(self, df, target_column, n_steps_in, n_steps_out):
        """Preprocesar los datos y dividirlos en secuencias."""
        # Convertir SETTLEMENTDATE a datetime
        df['SETTLEMENTDATE'] = pd.to_datetime(df['SETTLEMENTDATE'])

        # Extraer características temporales
        df['HOUR'] = df['SETTLEMENTDATE'].dt.hour
        df['DAYOFWEEK'] = df['SETTLEMENTDATE'].dt.dayofweek
        df['MONTH'] = df['SETTLEMENTDATE'].dt.month

        # Seleccionar características y objetivo
        features = ['HOUR', 'DAYOFWEEK', 'MONTH', 'RRP', 'PERIODTYPE']
        X = df[features]
        y = df[target_column]

        # Codificar PERIODTYPE (si es categórico)
        X = pd.get_dummies(X, columns=['PERIODTYPE'], drop_first=True)

        # Normalizar los datos
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Dividir la serie temporal en secuencias con múltiples pasos futuros
        X_seq, y_seq = self.split_sequence(X_scaled, y.values, n_steps_in, n_steps_out)

        # Dividir en conjuntos de entrenamiento y prueba
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_seq, y_seq, test_size=0.2, random_state=42
        )

        return self

    def run_pipeline(self, folder_path, target_column, n_steps_in, n_steps_out):
        """Ejecutar el pipeline completo."""
        df = self.load_data(folder_path)
        self.preprocessing(df, target_column, n_steps_in, n_steps_out)
        self.build_univariate_cnn()
        self.build_multivariate_cnn()
        self.build_multiple_header_cnn()
        #self.build_multiple_output_cnn() #error
        self.build_multiple_steps_cnn()
        self.evaluate_models(epochs=10)  # 10 épocas
        self.print_summary()  # Resumen final
        return self

