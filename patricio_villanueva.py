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

class PatricioSolution2:
    def __init__(self):
        """
        Initialize the PatricioSolution2 class with default values.
        """
        self.n_steps = None
        self.n_features = None
        self.X = None
        self.y = None

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
        
        self.mscnn = None
        self.history_mscnn = None
        
        self.mocnn = None
        self.history_mocnn = None

    def plot(self, df, title='Demand per Day', ylabel='Total Demand'):
        """
        Plot time series data.
        
        Parameters:
        df : pandas.Series
            The time series data to plot.
        title : str
            The title of the plot.
        ylabel : str
            The label for y-axis.
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

    def preprocessing(self, path):
        """
        Preprocess electricity data from multiple CSV files.
        
        Parameters:
        path : str
            The path to the directory containing CSV files.
            
        Returns:
        self : PatricioSolution2
            Returns the instance for method chaining.
        """
        # Load and concatenate all CSV files in the directory
        csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
        df_list = []
        
        for file in csv_files:
            try:
                temp_df = pd.read_csv(os.path.join(path, file))
                # Ensure SETTLEMENTDATE is treated as a datetime
                if 'SETTLEMENTDATE' in temp_df.columns:
                    temp_df['SETTLEMENTDATE'] = pd.to_datetime(temp_df['SETTLEMENTDATE'])
                df_list.append(temp_df)
            except Exception as e:
                print(f"Error loading file {file}: {e}")
        
        if not df_list:
            raise ValueError(f"No valid CSV files found in {path}")
            
        df = pd.concat(df_list, ignore_index=True)
        
        # Ensure SETTLEMENTDATE is set as the index and is datetime type
        if 'SETTLEMENTDATE' in df.columns:
            df.set_index('SETTLEMENTDATE', inplace=True)
        else:
            raise ValueError("SETTLEMENTDATE column not found in the dataset")
            
        # Filter data from 2021 onwards
        df = df[df.index >= '2021-01-01']
        
        # Resample to daily frequency
        if 'TOTALDEMAND' in df.columns:
            df2 = df['TOTALDEMAND'].resample('1D').mean()
            self.df2 = df2
            self.plot(df2)
        else:
            raise ValueError("TOTALDEMAND column not found in the dataset")
            
        # Store RRP data for multivariate analysis if available
        if 'RRP' in df.columns:
            self.df3 = df['RRP'].resample('1D').mean()
        
        return self

    @staticmethod
    def split_univariate_sequence(sequence, n_steps):
        """
        Split a univariate sequence into samples with a fixed window size.
        
        Parameters:
        sequence : array-like
            The input time series data.
        n_steps : int
            The window size.
            
        Returns:
        tuple
            X (input sequences) and y (target values).
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
        Set the features for the model and prepare the data.
        
        Parameters:
        n_steps : int
            The window size for the sequence.
        n_features : int
            The number of features in the data.
            
        Returns:
        self : PatricioSolution2
            Returns the instance for method chaining.
        """
        self.n_steps = n_steps
        self.n_features = n_features
        
        if self.df2 is None:
            raise ValueError("Data not preprocessed. Call preprocessing() first.")
            
        self.X, self.y = self.split_univariate_sequence(self.df2, self.n_steps)
        
        # Reshape input to be [samples, time steps, features]
        self.X = self.X.reshape((self.X.shape[0], self.X.shape[1], self.n_features))
        
        return self

    def univariate_cnn(self):
        """
        Build and train a univariate CNN model.
        
        Returns:
        self : PatricioSolution2
            Returns the instance for method chaining.
        """
        if self.X is None or self.y is None:
            raise ValueError("Features not set. Call set_features() first.")
            
        ucnn = Sequential()
        ucnn.add(keras.layers.Input(shape=(self.n_steps, self.n_features)))
        ucnn.add(Conv1D(64, 2, activation='relu'))
        ucnn.add(MaxPooling1D())
        ucnn.add(Flatten())
        ucnn.add(Dense(50, activation='relu'))
        ucnn.add(Dense(1))
        ucnn.compile(optimizer='adam', loss='mse')

        history_ucnn = ucnn.fit(self.X, self.y, epochs=1000, verbose=0)
        pd.DataFrame(history_ucnn.history).plot(title='Univariate CNN Training Loss')
        plt.grid(True)
        plt.show()

        self.ucnn = ucnn
        self.history_ucnn = history_ucnn
        
        # Evaluate the model
        self.evaluate(model=self.ucnn)

        return self

    def evaluate(self, model=None):
        """
        Evaluate a model on the test data.
        
        Parameters:
        model : keras model, optional
            The model to evaluate. If None, uses self.ucnn by default.
            
        Returns:
        self : PatricioSolution2
            Returns the instance for method chaining.
        """
        # Use the provided model or default to the univariate CNN model
        model_to_evaluate = model if model is not None else self.ucnn
        
        if model_to_evaluate is None:
            raise ValueError("No model available for evaluation")
            
        # Check if we're evaluating a model with multiple inputs
        if isinstance(model_to_evaluate, keras.models.Model) and len(model_to_evaluate.inputs) > 1:
            # For models like multiple header CNN that have multiple inputs
            if hasattr(self, 'X') and self.X is not None:
                X1 = self.X[:, :, 0].reshape(self.X.shape[0], self.X.shape[1], 1)
                X2 = self.X[:, :, 1].reshape(self.X.shape[0], self.X.shape[1], 1) if self.X.shape[2] > 1 else X1
                y_pred = model_to_evaluate.predict([X1, X2]).flatten()
                r2 = r2_score(self.y, y_pred)
                print(f'R² Score: {r2:.4f}')
            else:
                print("Warning: Cannot evaluate model with multiple inputs without proper data")
        elif hasattr(self, 'X_m') and model_to_evaluate == self.mpcnn:
            # For multiple parallel CNN
            y_pred = model_to_evaluate.predict(self.X_m)
            mse = np.mean((self.y_m - y_pred) ** 2)
            print(f'MSE: {mse:.4f}')
        else:
            # Standard evaluation for single input models
            if hasattr(self, 'X') and self.X is not None and hasattr(self, 'y') and self.y is not None:
                # Make predictions and evaluate
                y_pred = model_to_evaluate.predict(self.X).flatten()
                r2 = r2_score(self.y, y_pred)
                print(f'R² Score: {r2:.4f}')
            else:
                print("Warning: No data available for evaluation")

        return self

    def multivariate_cnn_preprocess(self):
        """
        Preprocess data for multivariate CNN.
        
        Returns:
        self : PatricioSolution2
            Returns the instance for method chaining.
        """
        if self.df2 is None or self.df3 is None:
            raise ValueError("Required data not available. Call preprocessing() first.")
            
        # Convert to numpy arrays and reshape
        df2_array = self.df2.to_numpy().reshape(-1, 1)
        df3_array = self.df3.to_numpy().reshape(-1, 1)
        
        # Ensure both arrays have the same length
        min_length = min(len(df2_array), len(df3_array))
        df2_array = df2_array[:min_length]
        df3_array = df3_array[:min_length]
        
        # Output sequence (could be any derived value)
        output_seq = (df2_array + df3_array).reshape(-1, 1)
        
        # Stack horizontally to create multivariate dataset [feature1, feature2, output]
        self.dataset = np.hstack([df2_array, df3_array, output_seq])
        
        return self

    @staticmethod
    def split_multivariate_sequence(sequence, n_steps):
        """
        Split a multivariate sequence into samples.
        
        Parameters:
        sequence : numpy.ndarray
            The multivariate sequence.
        n_steps : int
            The window size.
            
        Returns:
        tuple
            X (input sequences) and y (target values).
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

    def multivariate_cnn(self):
        """
        Build and train a multivariate CNN model.
        
        Returns:
        self : PatricioSolution2
            Returns the instance for method chaining.
        """
        if self.dataset is None:
            self.multivariate_cnn_preprocess()
            
        # Split the dataset
        X, y = self.split_multivariate_sequence(self.dataset, self.n_steps)
        
        n_features = X.shape[2]
        
        mcnn = Sequential()
        mcnn.add(keras.layers.Input(shape=(self.n_steps, n_features)))
        mcnn.add(Conv1D(64, 2, activation='relu'))
        mcnn.add(MaxPooling1D())
        mcnn.add(Flatten())
        mcnn.add(Dense(50, activation='relu'))
        mcnn.add(Dense(1))
        mcnn.compile(optimizer='adam', loss='mse')
        
        history_mcnn = mcnn.fit(X, y, epochs=1000, verbose=0)
        pd.DataFrame(history_mcnn.history).plot(title='Multivariate CNN Training Loss')
        plt.grid(True)
        plt.show()

        self.mcnn = mcnn
        self.history_mcnn = history_mcnn
        
        # Store data for evaluation
        self.X = X
        self.y = y
        
        # Evaluate the model
        self.evaluate(model=self.mcnn)

        return self

    def multiple_header(self):
        """
        Build and train a multiple header CNN model.
        
        Returns:
        self : PatricioSolution2
            Returns the instance for method chaining.
        """
        if self.dataset is None:
            self.multivariate_cnn_preprocess()
            
        # Split dataset and prepare data for multiple header model
        X, y = self.split_multivariate_sequence(self.dataset, self.n_steps)
        
        n_steps = X.shape[1]
        n_features = X.shape[2]

        self.n_steps = n_steps
        self.n_features = n_features

        # Define the model with two separate inputs
        model_input1 = Input(shape=(n_steps, 1))
        cnn1 = Conv1D(64, 2, activation='relu')(model_input1)
        cnn1 = MaxPooling1D()(cnn1)
        cnn1 = Flatten()(cnn1)

        model_input2 = Input(shape=(n_steps, 1))
        cnn2 = Conv1D(64, 2, activation='relu')(model_input2)
        cnn2 = MaxPooling1D()(cnn2)
        cnn2 = Flatten()(cnn2)

        # Merge the processed inputs
        final_model = concatenate([cnn1, cnn2])
        dense = Dense(50, activation='relu')(final_model)
        output = Dense(1)(dense)

        # Create the model
        from tensorflow.keras.models import Model
        mhcnn = Model(inputs=[model_input1, model_input2], outputs=output)
        mhcnn.compile(optimizer='adam', loss='mse')

        self.mhcnn = mhcnn
        
        # Prepare the data for the model inputs
        X1 = X[:, :, 0].reshape(X.shape[0], X.shape[1], 1)  
        X2 = X[:, :, 1].reshape(X.shape[0], X.shape[1], 1)  
        
        # Train the model
        history_mhcnn = self.mhcnn.fit([X1, X2], y, epochs=1000, verbose=0)
        pd.DataFrame(history_mhcnn.history).plot(title='Multiple Header CNN Training Loss')
        plt.grid(True)
        plt.show()

        self.history_mhcnn = history_mhcnn
        
        # Store data for evaluation
        self.X = X
        self.y = y
        
        # Evaluate the model
        loss = self.mhcnn.evaluate([X1, X2], y)
        print(f'Loss: {loss:.4f}')

        return self

    @staticmethod
    def split_multiple_forecasting_sequence(sequence, n_steps):
        """
        Split a sequence for multiple forecasting.
        
        Parameters:
        sequence : numpy.ndarray
            The multivariate sequence.
        n_steps : int
            The window size.
            
        Returns:
        tuple
            X (input sequences) and y (target sequences).
        """
        X, y = [], []
        for i in range(len(sequence)):
            end_ix = i + n_steps
            
            if end_ix > len(sequence) - 1:
                break
            # Input is the window, output is the next value for all features
            seq_x, seq_y = sequence[i:end_ix, :], sequence[end_ix, :]
            X.append(seq_x)
            y.append(seq_y)

        return np.array(X), np.array(y)

    def multiple_parallel(self):
        """
        Build and train a multiple parallel CNN model.
        
        Returns:
        self : PatricioSolution2
            Returns the instance for method chaining.
        """
        if self.dataset is None:
            self.multivariate_cnn_preprocess()
            
        # Split the dataset
        X_m, y_m = self.split_multiple_forecasting_sequence(self.dataset, n_steps=4)
        
        self.X_m = X_m
        self.y_m = y_m
        
        n_features = X_m.shape[2]

        mpcnn = Sequential()
        mpcnn.add(keras.layers.Input(shape=(self.n_steps, n_features)))
        mpcnn.add(Conv1D(64, 2, activation='relu'))
        mpcnn.add(MaxPooling1D())
        mpcnn.add(Flatten())
        mpcnn.add(Dense(50, activation='relu'))
        mpcnn.add(Dense(n_features))  # Output for all features
        mpcnn.compile(optimizer='adam', loss='mse')

        history_mpcnn = mpcnn.fit(X_m, y_m, epochs=1000, verbose=0)
        pd.DataFrame(history_mpcnn.history).plot(title='Multiple Parallel CNN Training Loss')
        plt.grid(True)
        plt.show()

        self.mpcnn = mpcnn
        self.history_mpcnn = history_mpcnn

        # Evaluate the model
        loss = self.mpcnn.evaluate(X_m, y_m)
        print(f'Loss: {loss:.4f}')

        return self

    def multi_output_cnn(self):
        """
        Build and train a multi-output CNN model.
        
        Returns:
        self : PatricioSolution2
            Returns the instance for method chaining.
        """
        if self.X_m is None or self.y_m is None:
            if self.dataset is None:
                self.multivariate_cnn_preprocess()
            self.X_m, self.y_m = self.split_multiple_forecasting_sequence(self.dataset, n_steps=4)
            
        n_features = self.X_m.shape[2]

        # Define the model
        from tensorflow.keras.models import Model
        visible = Input(shape=(self.n_steps, n_features))
        cnn = Conv1D(64, 2, activation='relu')(visible)
        cnn = MaxPooling1D()(cnn)
        cnn = Flatten()(cnn)
        cnn = Dense(50, activation='relu')(cnn)

        # Separate output for each feature
        output1 = Dense(1)(cnn)
        output2 = Dense(1)(cnn)
        output3 = Dense(1)(cnn)
        
        # Create the model
        mocnn = Model(inputs=visible, outputs=[output1, output2, output3])
        mocnn.compile(optimizer='adam', loss='mse')

        # Prepare target outputs
        y1 = self.y_m[:, 0].reshape((self.y_m.shape[0], 1))
        y2 = self.y_m[:, 1].reshape((self.y_m.shape[0], 1))
        y3 = self.y_m[:, 2].reshape((self.y_m.shape[0], 1))

        # Train the model
        history_mocnn = mocnn.fit(self.X_m, [y1, y2, y3], epochs=1000, verbose=0)
        pd.DataFrame(history_mocnn.history).plot(title='Multi-Output CNN Training Loss')
        plt.grid(True)
        plt.show()

        self.mocnn = mocnn
        self.history_mocnn = history_mocnn

        # Evaluate the model
        loss = self.mocnn.evaluate(self.X_m, [y1, y2, y3])
        print(f'Loss: {loss}')

        return self

    @staticmethod
    def split_univariate_sequence_m_step(sequence, n_steps_in, n_steps_out):
        """
        Split a univariate sequence into samples with multiple output steps.
        
        Parameters:
        sequence : array-like
            The input time series data.
        n_steps_in : int
            The input window size.
        n_steps_out : int
            The output window size.
            
        Returns:
        tuple
            X (input sequences) and y (output sequences).
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

    def multiple_steps_cnn(self):
        """
        Build and train a multiple steps CNN model.
        
        Returns:
        self : PatricioSolution2
            Returns the instance for method chaining.
        """
        if self.df2 is None:
            raise ValueError("Data not preprocessed. Call preprocessing() first.")
            
        # Split data for multiple step prediction
        X, y = self.split_univariate_sequence_m_step(self.df2, 4, 2)
        
        # Display a few examples
        for i in range(min(3, len(X))):
            print(f"Input {i}: {X[i]}, Output {i}: {y[i]}")
        
        # Reshape for CNN
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Build the model
        mscnn = Sequential()
        mscnn.add(keras.layers.Input(shape=(4, 1)))
        mscnn.add(Conv1D(64, 2, activation='relu'))
        mscnn.add(MaxPooling1D())
        mscnn.add(Flatten())
        mscnn.add(Dense(50, activation='relu'))
        mscnn.add(Dense(2))  # Output 2 future values
        mscnn.compile(optimizer='adam', loss='mse')

        # Train the model
        history_mscnn = mscnn.fit(X, y, epochs=1000, verbose=0)
        pd.DataFrame(history_mscnn.history).plot(title='Multiple Steps CNN Training Loss')
        plt.grid(True)
        plt.show()

        self.mscnn = mscnn
        self.history_mscnn = history_mscnn

        # Evaluate the model
        loss = self.mscnn.evaluate(X, y)
        print(f'Loss: {loss:.4f}')

        return self

    def execute_activity_two(self, path):
        """
        Execute all models for activity two.
        
        Parameters:
        path : str
            The path to the directory containing CSV files.
            
        Returns:
        self : PatricioSolution2
            Returns the instance for method chaining.
        """
        print("Starting preprocessing...")
        self.preprocessing(path)
        
        print("Setting up features...")
        self.set_features(n_steps=4, n_features=1)
        
        print("Training univariate CNN model...")
        self.univariate_cnn()
        
        print("Preprocessing data for multivariate analysis...")
        self.multivariate_cnn_preprocess()
        
        print("Training multivariate CNN model...")
        self.multivariate_cnn()
        
        print("Training multiple header CNN model...")
        self.multiple_header()
        
        print("Training multiple parallel CNN model...")
        self.multiple_parallel()
        
        print("Training multi-output CNN model...")
        self.multi_output_cnn()
        
        print("Training multiple steps CNN model...")
        self.multiple_steps_cnn()
        
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
