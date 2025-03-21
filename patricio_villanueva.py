# Homework 4 Patricio villanueva class for solution

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Dropout, MaxPooling1D
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Input, concatenate
from keras.utils import plot_model

import os

class PatricioSolution1:
    def __init__(self):
        """
        Initialize the solution with default values.
        """
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.fcm = None  
        self.cm = None   
        self.history = None
        self.dataset = None

    # Activity one
    def preprocessing(self, df):
        """
        Preprocess the data.
        
        Parameters:
        df : pandas.DataFrame
            The input data.
            
        Returns:
        self : PatricioSolution1
            Returns the instance for method chaining.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
            
        if 'price' not in df.columns:
            raise ValueError("DataFrame must contain a 'price' column")
            
        X = df.drop(columns=['price'])
        y = df['price']

        categorical_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
                          'airconditioning', 'prefarea', 'furnishingstatus']
        # Ensure all categorical columns exist in the dataset
        for col in categorical_cols:
            if col not in X.columns:
                raise ValueError(f"Column '{col}' not found in the dataset")
                
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        X[numerical_cols] = X[numerical_cols].fillna(X[numerical_cols].median())

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(), categorical_cols)
            ])
        X_processed = preprocessor.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        return self 

    def fully_connected_model(self):
        """
        Build a fully connected model.
        
        Returns:
        self : PatricioSolution1
            Returns the instance for method chaining.
        """
        if self.X_train is None:
            raise ValueError("Data not preprocessed. Call preprocessing() first.")
            
        # Create the model architecture
        fcm = Sequential([
            Dense(128, activation='relu', input_shape=(self.X_train.shape[1],)),  
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1)
        ])

        # Compile the model
        fcm.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Display model summary
        print("Fully Connected Model Summary:")
        fcm.summary()

        self.fcm = fcm

        return self

    def evaluate(self, model=None):
        """
        Evaluate a model on the test data.
        
        Parameters:
        model : keras model, optional
            The model to evaluate. If None, uses self.fcm by default.
            
        Returns:
        self : PatricioSolution1
            Returns the instance for method chaining.
        """
        # Validate data is available
        if self.X_train is None or self.X_test is None:
            raise ValueError("Data not preprocessed. Call preprocessing() first.")
            
        # Use the provided model or default to the fully connected model
        model_to_evaluate = model if model is not None else self.fcm
        
        if model_to_evaluate is None:
            raise ValueError("No model available to evaluate. Build a model first.")
        
        # Train the model
        print(f"Training model with {len(self.X_train)} samples...")
        history = model_to_evaluate.fit(
            self.X_train, self.y_train, 
            epochs=400, batch_size=16, 
            validation_data=(self.X_test, self.y_test),
            verbose=1
        )
        
        # Make predictions and evaluate
        print("Evaluating model on test data...")
        y_pred = model_to_evaluate.predict(self.X_test).flatten()
        r2 = r2_score(self.y_test, y_pred)
        print(f'R² Score: {r2:.4f}')
        
        # Plot training history
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title('Mean Absolute Error Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

        self.history = history

        return self

    def convolutional_model(self):
        """
        Build a convolutional model.
        
        Returns:
        self : PatricioSolution1
            Returns the instance for method chaining.
        """
        if self.X_train is None:
            raise ValueError("Data not preprocessed. Call preprocessing() first.")
        
        # Reshape data for CNN (add channel dimension)
        X_train = self.X_train.reshape(self.X_train.shape[0], self.X_train.shape[1], 1)
        X_test = self.X_test.reshape(self.X_test.shape[0], self.X_test.shape[1], 1)

        # Build the CNN model
        cm = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])

        # Compile the model
        cm.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Display model summary
        print("Convolutional Model Summary:")
        cm.summary()

        self.cm = cm
        self.X_train = X_train  # Store reshaped data
        self.X_test = X_test

        return self

    def execute_activity_one(self, df):
        """
        Execute all models for activity one.
        
        Parameters:
        df : pandas.DataFrame
            The input data.
            
        Returns:
        self : PatricioSolution1
            Returns the instance for method chaining.
        """
        print("Step 1: Preprocessing data...")
        self.preprocessing(df)
        
        print("\nStep 2: Building and evaluating fully connected model...")
        self.fully_connected_model()
        self.evaluate()
        
        print("\nStep 3: Building and evaluating convolutional model...")
        self.convolutional_model()
        self.evaluate(model=self.cm)
        
        print("\nAll models completed!")
        return self

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Bidirectional, LSTM, concatenate
from sklearn.metrics import r2_score

class PatricioSolution2:
    def __init__(self):
        """
        Initialize the class with default values
        """
        # Datos y configuraciones generales
        self.n_steps = None
        self.n_features = None
        self.X = None
        self.y = None

        # Modelos y sus historiales
        self.ucnn = None                   
        self.history_ucnn = None

        self.df2 = None                     
        self.df3 = None                     
        self.dataset = None                 
        
        self.mcnn = None                    
        self.history_mcnn = None

        self.mhcnn = None                   
        self.history_mhcnn = None

        self.X_m = None                   
        self.y_m = None

        self.mpcnn = None                   
        self.history_mpcnn = None
        
        self.mocnn = None                   
        self.history_mocnn = None

        self.mscnn = None                   
        self.history_mscnn = None

        self.ucnn_dropout = None            
        self.history_ucnn_dropout = None
        
        self.ucnn_bidirectional = None      
        self.history_ucnn_bidirectional = None
        
        self.ulstm = None                   
        self.history_ulstm = None
        
        self.udense = None                  
        self.history_udense = None
        
        self.ucnn_stack = None              
        self.history_ucnn_stack = None
        
        self.ucnn_ensemble = None           
        self.history_ucnn_ensemble = {}     

    def plot(self, df, title='Demand per Day', ylabel='Total Demand'):
        """
        Graph a time series.

        Parameters:
        df : pandas.DataFrame
            The time series data.
            
        title : str, optional
            The title of the plot. Defaults to 'Demand per Day'.
            
        ylabel : str, optional
            The label for the y-axis. Defaults to 'Total Demand'.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(df.index, df, marker='o', linestyle='-')
        plt.xlabel('Time')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_real_vs_predicted(self, X, y, model, model_name="Model"):
        """
        Graph the real series against the model predictions.
        
        Parameters:
        X : array-like
            The input data.
            
        y : array-like
            The target values.
            
        model : keras model
            The model to evaluate.
            
        model_name : str, optional
            The name of the model. Defaults to 'Model'.
        """
        y_pred = model.predict(X).flatten()
        plt.figure(figsize=(10, 5))
        plt.plot(y, label='Real')
        plt.plot(y_pred, label='Prediction')
        plt.title(f"{model_name}: Real vs Predicho")
        plt.xlabel("Ejemplo")
        plt.ylabel("Valor")
        plt.legend()
        plt.grid(True)
        plt.show()

    def preprocessing(self, path):
        """
        Preprocess electric data from multiple CSV files.
        
        Parameters:
        path : str
            The path to the directory containing the CSV files.
        """
        csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
        df_list = []
        for file in csv_files:
            try:
                temp_df = pd.read_csv(os.path.join(path, file))
                if 'SETTLEMENTDATE' in temp_df.columns:
                    temp_df['SETTLEMENTDATE'] = pd.to_datetime(temp_df['SETTLEMENTDATE'])
                df_list.append(temp_df)
            except Exception as e:
                print(f"Error loading file {file}: {e}")
        if not df_list:
            raise ValueError(f"No valid CSV files found in {path}")
        df = pd.concat(df_list, ignore_index=True)
        if 'SETTLEMENTDATE' in df.columns:
            df.set_index('SETTLEMENTDATE', inplace=True)
        else:
            raise ValueError("SETTLEMENTDATE column not found in the dataset")
        df = df[df.index >= '2021-01-01']
        if 'TOTALDEMAND' in df.columns:
            df2 = df['TOTALDEMAND'].resample('1D').mean()
            self.df2 = df2
            self.plot(df2)
        else:
            raise ValueError("TOTALDEMAND column not found in the dataset")
        if 'RRP' in df.columns:
            self.df3 = df['RRP'].resample('1D').mean()
        return self

    @staticmethod
    def split_univariate_sequence(sequence, n_steps):
        """
        Splits a univariate sequence into windows of size n_steps.
        
        Parameters:
        sequence : array-like
            The sequence to split.
            
        n_steps : int
            The size of each window.
            
        Returns:
        X : numpy.ndarray
            The input data.
            
        y : numpy.ndarray
            The target values.
        """
        X, y = [], []
        for i in range(len(sequence)):
            end_ix = i + n_steps
            if end_ix > len(sequence) - 1:
                break
            if isinstance(sequence, pd.Series):
                seq_x, seq_y = sequence.iloc[i:end_ix], sequence.iloc[end_ix]
            else:
                seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)

    def set_features(self, n_steps=4, n_features=1):
        """
        Prepare features for the univariate model.
        
        Parameters:
        n_steps : int, optional
            The number of time steps to use for the model. Defaults to 4.
            
        n_features : int, optional
            The number of features to use for the model. Defaults to 1.
        """
        self.n_steps = n_steps
        self.n_features = n_features
        if self.df2 is None:
            raise ValueError("Data not preprocessed. Call preprocessing() first.")
        self.X, self.y = self.split_univariate_sequence(self.df2, self.n_steps)
        self.X = self.X.reshape((self.X.shape[0], self.X.shape[1], self.n_features))
        return self

    def evaluate(self, model=None):
        """
        Evaluate a model (predict and show R² or MSE) and graph real vs predicted.
        
        Parameters:
        model : keras model, optional
            The model to evaluate. If None, uses self.ucnn by default.
        """
        model_to_evaluate = model if model is not None else self.ucnn
        if model_to_evaluate is None:
            raise ValueError("No model available for evaluation")
        # Evaluación para modelos de entrada única
        if hasattr(self, 'X') and self.X is not None and hasattr(self, 'y') and self.y is not None:
            y_pred = model_to_evaluate.predict(self.X).flatten()
            r2 = r2_score(self.y, y_pred)
            print(f'R² Score: {r2:.4f}')
            self.plot_real_vs_predicted(self.X, self.y, model_to_evaluate, model_name=model_to_evaluate.name)
        else:
            print("Warning: No data available for evaluation")
        return self

    def univariate_cnn(self):
        """
        Univariate CNN model.
        
        Returns:
        self : PatricioSolution2
            Returns the instance for method chaining.
        """
        if self.X is None or self.y is None:
            raise ValueError("Features not set. Call set_features() first.")
        model = Sequential(name="Univariate_CNN")
        model.add(keras.layers.Input(shape=(self.n_steps, self.n_features)))
        model.add(Conv1D(64, 2, activation='relu'))
        model.add(MaxPooling1D())
        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        history = model.fit(self.X, self.y, epochs=1000, verbose=0)
        pd.DataFrame(history.history).plot(title='Univariate CNN Training Loss')
        plt.grid(True)
        plt.show()
        self.ucnn = model
        self.history_ucnn = history
        self.evaluate(model=model)
        return self

    def multivariate_cnn(self):
        """
        Multivariate CNN model.
        
        Returns:
        self : PatricioSolution2
            Returns the instance for method chaining.
        """
        if self.dataset is None:
            self.multivariate_cnn_preprocess()
        X, y = self.split_multivariate_sequence(self.dataset, self.n_steps)
        n_features = X.shape[2]
        model = Sequential(name="Multivariate_CNN")
        model.add(keras.layers.Input(shape=(self.n_steps, n_features)))
        model.add(Conv1D(64, 2, activation='relu'))
        model.add(MaxPooling1D())
        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        history = model.fit(X, y, epochs=1000, verbose=0)
        pd.DataFrame(history.history).plot(title='Multivariate CNN Training Loss')
        plt.grid(True)
        plt.show()
        self.mcnn = model
        self.history_mcnn = history
        # Guarda X e y para evaluación
        self.X, self.y = X, y
        self.evaluate(model=model)
        return self

    def multiple_header(self):
        """
        Multiple Header CNN (multi-input) model.
        
        Returns:
        self : PatricioSolution2
            Returns the instance for method chaining.
        """
        if self.dataset is None:
            self.multivariate_cnn_preprocess()
        X, y = self.split_multivariate_sequence(self.dataset, self.n_steps)
        n_steps_local = X.shape[1]
        n_features_local = X.shape[2]
        self.n_steps = n_steps_local
        self.n_features = n_features_local
        input1 = Input(shape=(n_steps_local, 1))
        cnn1 = Conv1D(64, 2, activation='relu')(input1)
        cnn1 = MaxPooling1D()(cnn1)
        cnn1 = Flatten()(cnn1)
        input2 = Input(shape=(n_steps_local, 1))
        cnn2 = Conv1D(64, 2, activation='relu')(input2)
        cnn2 = MaxPooling1D()(cnn2)
        cnn2 = Flatten()(cnn2)
        merged = concatenate([cnn1, cnn2])
        dense = Dense(50, activation='relu')(merged)
        output = Dense(1)(dense)
        model = Model(inputs=[input1, input2], outputs=output, name="Multiple_Header_CNN")
        model.compile(optimizer='adam', loss='mse')
        X1 = X[:, :, 0].reshape(X.shape[0], n_steps_local, 1)
        X2 = X[:, :, 1].reshape(X.shape[0], n_steps_local, 1)
        history = model.fit([X1, X2], y, epochs=1000, verbose=0)
        pd.DataFrame(history.history).plot(title='Multiple Header CNN Training Loss')
        plt.grid(True)
        plt.show()
        self.mhcnn = model
        self.history_mhcnn = history
        self.X, self.y = X, y
        loss = model.evaluate([X1, X2], y)
        print(f'Loss: {loss:.4f}')
        return self

    def multiple_parallel(self):
        """
        Multiple Parallel CNN model.
        
        Returns:
        self : PatricioSolution2
            Returns the instance for method chaining.
        """
        if self.dataset is None:
            self.multivariate_cnn_preprocess()
        X_m, y_m = self.split_multiple_forecasting_sequence(self.dataset, n_steps=4)
        self.X_m, self.y_m = X_m, y_m
        n_features_local = X_m.shape[2]
        model = Sequential(name="Multiple_Parallel_CNN")
        model.add(keras.layers.Input(shape=(self.n_steps, n_features_local)))
        model.add(Conv1D(64, 2, activation='relu'))
        model.add(MaxPooling1D())
        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        model.add(Dense(n_features_local))
        model.compile(optimizer='adam', loss='mse')
        history = model.fit(X_m, y_m, epochs=1000, verbose=0)
        pd.DataFrame(history.history).plot(title='Multiple Parallel CNN Training Loss')
        plt.grid(True)
        plt.show()
        self.mpcnn = model
        self.history_mpcnn = history
        loss = model.evaluate(X_m, y_m)
        print(f'Loss: {loss:.4f}')
        return self

    def multi_output_cnn(self):
        """
        Multi-Output CNN model.
        
        Returns:
        self : PatricioSolution2
            Returns the instance for method chaining.
        """
        if self.X_m is None or self.y_m is None:
            if self.dataset is None:
                self.multivariate_cnn_preprocess()
            self.X_m, self.y_m = self.split_multiple_forecasting_sequence(self.dataset, n_steps=4)
        n_features_local = self.X_m.shape[2]
        visible = Input(shape=(self.n_steps, n_features_local))
        cnn = Conv1D(64, 2, activation='relu')(visible)
        cnn = MaxPooling1D()(cnn)
        cnn = Flatten()(cnn)
        cnn = Dense(50, activation='relu')(cnn)
        output1 = Dense(1)(cnn)
        output2 = Dense(1)(cnn)
        output3 = Dense(1)(cnn)
        model = Model(inputs=visible, outputs=[output1, output2, output3], name="Multi_Output_CNN")
        model.compile(optimizer='adam', loss='mse')
        y1 = self.y_m[:, 0].reshape((self.y_m.shape[0], 1))
        y2 = self.y_m[:, 1].reshape((self.y_m.shape[0], 1))
        y3 = self.y_m[:, 2].reshape((self.y_m.shape[0], 1))
        history = model.fit(self.X_m, [y1, y2, y3], epochs=1000, verbose=0)
        pd.DataFrame(history.history).plot(title='Multi-Output CNN Training Loss')
        plt.grid(True)
        plt.show()
        self.mocnn = model
        self.history_mocnn = history
        loss = model.evaluate(self.X_m, [y1, y2, y3])
        print(f'Loss: {loss}')
        return self

    def multiple_steps_cnn(self):
        """
        Multiple Steps CNN model.
        
        Returns:
        self : PatricioSolution2
            Returns the instance for method chaining.
        """
        if self.df2 is None:
            raise ValueError("Data not preprocessed. Call preprocessing() first.")
        X, y = self.split_univariate_sequence_m_step(self.df2, 4, 2)
        for i in range(min(3, len(X))):
            print(f"Input {i}: {X[i]}, Output {i}: {y[i]}")
        X = X.reshape((X.shape[0], X.shape[1], 1))
        model = Sequential(name="Multiple_Steps_CNN")
        model.add(keras.layers.Input(shape=(4, 1)))
        model.add(Conv1D(64, 2, activation='relu'))
        model.add(MaxPooling1D())
        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        model.add(Dense(2))
        model.compile(optimizer='adam', loss='mse')
        history = model.fit(X, y, epochs=1000, verbose=0)
        pd.DataFrame(history.history).plot(title='Multiple Steps CNN Training Loss')
        plt.grid(True)
        plt.show()
        self.mscnn = model
        self.history_mscnn = history
        loss = model.evaluate(X, y)
        print(f'Loss: {loss:.4f}')
        return self

    def univariate_cnn_dropout(self):
        """
        Univariate CNN with Dropout model.
        
        Returns:
        self : PatricioSolution2
            Returns the instance for method chaining.
        """
        if self.X is None or self.y is None:
            raise ValueError("Features not set. Call set_features() first.")
        model = Sequential(name="Univariate_CNN_Dropout")
        model.add(keras.layers.Input(shape=(self.n_steps, self.n_features)))
        model.add(Conv1D(64, 2, activation='relu'))
        model.add(Dropout(0.2))
        model.add(MaxPooling1D())
        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        history = model.fit(self.X, self.y, epochs=1000, verbose=0)
        pd.DataFrame(history.history).plot(title='Univariate CNN Dropout Training Loss')
        plt.grid(True)
        plt.show()
        self.ucnn_dropout = model
        self.history_ucnn_dropout = history
        self.evaluate(model=model)
        return self

    def univariate_cnn_bidirectional(self):
        """
        Univariate CNN with Bidirectional LSTM model.
        
        Returns:
        self : PatricioSolution2
            Returns the instance for method chaining.
        """
        if self.X is None or self.y is None:
            raise ValueError("Features not set. Call set_features() first.")
        model = Sequential(name="Univariate_CNN_Bidirectional")
        model.add(keras.layers.Input(shape=(self.n_steps, self.n_features)))
        model.add(Conv1D(64, 2, activation='relu'))
        model.add(MaxPooling1D())
        model.add(Bidirectional(LSTM(50)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        history = model.fit(self.X, self.y, epochs=1000, verbose=0)
        pd.DataFrame(history.history).plot(title='Univariate CNN Bidirectional Training Loss')
        plt.grid(True)
        plt.show()
        self.ucnn_bidirectional = model
        self.history_ucnn_bidirectional = history
        self.evaluate(model=model)
        return self

    def univariate_lstm(self):
        """
        Univariate LSTM model.
        
        Returns:
        self : PatricioSolution2
            Returns the instance for method chaining.
        """
        if self.X is None or self.y is None:
            raise ValueError("Features not set. Call set_features() first.")
        model = Sequential(name="Univariate_LSTM")
        model.add(keras.layers.Input(shape=(self.n_steps, self.n_features)))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        history = model.fit(self.X, self.y, epochs=1000, verbose=0)
        pd.DataFrame(history.history).plot(title='Univariate LSTM Training Loss')
        plt.grid(True)
        plt.show()
        self.ulstm = model
        self.history_ulstm = history
        self.evaluate(model=model)
        return self

    def univariate_dense(self):
        """
        Univariate Dense model.
        
        Returns:
        self : PatricioSolution2
            Returns the instance for method chaining.
        """
        if self.X is None or self.y is None:
            raise ValueError("Features not set. Call set_features() first.")
        model = Sequential(name="Univariate_Dense")
        model.add(keras.layers.Input(shape=(self.n_steps, self.n_features)))
        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        history = model.fit(self.X, self.y, epochs=1000, verbose=0)
        pd.DataFrame(history.history).plot(title='Univariate Dense Training Loss')
        plt.grid(True)
        plt.show()
        self.udense = model
        self.history_udense = history
        self.evaluate(model=model)
        return self

    def univariate_cnn_stack(self):
        """
        Stacked Univariate CNN (dos capas Conv1D) model.
        
        Returns:
        self : PatricioSolution2
            Returns the instance for method chaining.
        """
        if self.X is None or self.y is None:
            raise ValueError("Features not set. Call set_features() first.")
        model = Sequential(name="Univariate_CNN_Stack")
        model.add(keras.layers.Input(shape=(self.n_steps, self.n_features)))
        model.add(Conv1D(64, 2, activation='relu'))
        model.add(Conv1D(64, 2, activation='relu'))
        model.add(MaxPooling1D())
        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        history = model.fit(self.X, self.y, epochs=1000, verbose=0)
        pd.DataFrame(history.history).plot(title='Stacked Univariate CNN Training Loss')
        plt.grid(True)
        plt.show()
        self.ucnn_stack = model
        self.history_ucnn_stack = history
        self.evaluate(model=model)
        return self

    def univariate_cnn_ensemble(self):
        """
        Ensamble de dos modelos univariados (Univariate_CNN y Univariate_CNN_Dropout).        
        Returns:
        self : PatricioSolution2
            Returns the instance for method chaining.
        """
        if self.X is None or self.y is None:
            raise ValueError("Features not set. Call set_features() first.")

        model1 = Sequential(name="Ensemble_Model_1")
        model1.add(keras.layers.Input(shape=(self.n_steps, self.n_features)))
        model1.add(Conv1D(64, 2, activation='relu'))
        model1.add(MaxPooling1D())
        model1.add(Flatten())
        model1.add(Dense(50, activation='relu'))
        model1.add(Dense(1))
        model1.compile(optimizer='adam', loss='mse')
        history1 = model1.fit(self.X, self.y, epochs=1000, verbose=0)

        model2 = Sequential(name="Ensemble_Model_2")
        model2.add(keras.layers.Input(shape=(self.n_steps, self.n_features)))
        model2.add(Conv1D(64, 2, activation='relu'))
        model2.add(Dropout(0.2))
        model2.add(MaxPooling1D())
        model2.add(Flatten())
        model2.add(Dense(50, activation='relu'))
        model2.add(Dense(1))
        model2.compile(optimizer='adam', loss='mse')
        history2 = model2.fit(self.X, self.y, epochs=1000, verbose=0)

        self.history_ucnn_ensemble = {"model1": history1, "model2": history2}

        def ensemble_predict(X_input):
            pred1 = model1.predict(X_input).flatten()
            pred2 = model2.predict(X_input).flatten()
            return (pred1 + pred2) / 2.0

        y_pred = ensemble_predict(self.X)
        r2 = r2_score(self.y, y_pred)
        print(f'Ensemble Model R² Score: {r2:.4f}')
        plt.figure(figsize=(10, 5))
        plt.plot(self.y, label='Real')
        plt.plot(y_pred, label='Predicción Ensamble')
        plt.title("Ensemble: Real vs Predicho")
        plt.xlabel("Ejemplo")
        plt.ylabel("Valor")
        plt.legend()
        plt.grid(True)
        plt.show()
        self.ucnn_ensemble = ensemble_predict
        return self


    def multivariate_cnn_preprocess(self):
        """
        Preprocess data for multivariate analysis.
        
        Returns:
        self : PatricioSolution2
            Returns the instance for method chaining.
        """
        if self.df2 is None or self.df3 is None:
            raise ValueError("Required data not available. Call preprocessing() first.")
        df2_array = self.df2.to_numpy().reshape(-1, 1)
        df3_array = self.df3.to_numpy().reshape(-1, 1)
        min_length = min(len(df2_array), len(df3_array))
        df2_array = df2_array[:min_length]
        df3_array = df3_array[:min_length]
        output_seq = (df2_array + df3_array).reshape(-1, 1)
        self.dataset = np.hstack([df2_array, df3_array, output_seq])
        return self

    @staticmethod
    def split_multivariate_sequence(sequence, n_steps):
        """
        Splits a multivariate sequence into samples.
        
        Returns:
        X : numpy.ndarray
            Input features.
        y : numpy.ndarray
            Output values.
        """
        X, y = [], []
        for i in range(len(sequence)):
            end_ix = i + n_steps
            if end_ix > len(sequence):
                break
            seq_x, seq_y = sequence[i:end_ix, :-1], sequence[end_ix - 1, -1]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)

    @staticmethod
    def split_multiple_forecasting_sequence(sequence, n_steps):
        """
        Splits a sequence for multiple-step forecasting.
        
        Returns:
        X : numpy.ndarray
            Input features.
        y : numpy.ndarray
            Output values.
        """
        X, y = [], []
        for i in range(len(sequence)):
            end_ix = i + n_steps
            if end_ix > len(sequence) - 1:
                break
            seq_x, seq_y = sequence[i:end_ix, :], sequence[end_ix, :]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)

    @staticmethod
    def split_univariate_sequence_m_step(sequence, n_steps_in, n_steps_out):
        """
        Splits a univariate sequence into samples with multiple output steps.
        
        Returns:
        X : numpy.ndarray
            Input features.
        y : numpy.ndarray
            Output values.
        """
        X, y = [], []
        for i in range(len(sequence)):
            end_ix = i + n_steps_in
            out_end_ix = end_ix + n_steps_out
            if out_end_ix > len(sequence):
                break
            if isinstance(sequence, pd.Series):
                seq_x = sequence.iloc[i:end_ix].values
                seq_y = sequence.iloc[end_ix:out_end_ix].values
            else:
                seq_x = sequence[i:end_ix]
                seq_y = sequence[end_ix:out_end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)

    def execute_activity_two(self, path):
        """
        Execute all models (12 in total) for activity two.
        
        Parameters:
        path : str
            Path to the dataset.
        
        Returns:
        self : PatricioSolution2
            Returns the instance for method chaining.
        """
        print("Starting preprocessing...")
        self.preprocessing(path)
        print("Setting up features...")
        self.set_features(n_steps=4, n_features=1)
        print("Training Univariate CNN model...")
        self.univariate_cnn()
        print("Training Multivariate CNN model...")
        self.multivariate_cnn()
        print("Training Multiple Header CNN model...")
        self.multiple_header()
        print("Training Multiple Parallel CNN model...")
        self.multiple_parallel()
        print("Training Multi-Output CNN model...")
        self.multi_output_cnn()
        print("Training Multiple Steps CNN model...")
        self.multiple_steps_cnn()
        print("Training Univariate CNN with Dropout model...")
        self.univariate_cnn_dropout()
        print("Training Univariate CNN with Bidirectional LSTM model...")
        self.univariate_cnn_bidirectional()
        print("Training Univariate LSTM model...")
        self.univariate_lstm()
        print("Training Univariate Dense model...")
        self.univariate_dense()
        print("Training Stacked Univariate CNN model...")
        self.univariate_cnn_stack()
        print("Training Ensemble Univariate CNN model...")
        self.univariate_cnn_ensemble()
        print("All models completed!")
        return self


if __name__ == '__main__':
    print("Activity 1 main execution")
    df = pd.read_csv('data/Housing.csv')
    patricio_solution = PatricioSolution1()
    patricio_solution.execute_activity_one(df)

    print("Activity 2 main execution")
    path = 'data/electricity'
    patricio_solution2 = PatricioSolution2()
    patricio_solution2.execute_activity_two(path)
