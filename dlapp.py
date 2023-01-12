#!/usr/bin/env python
# coding: utf-8

# ## Jena Climate dataset
# 
# For this study i have used <a href="https://www.bgc-jena.mpg.de/wetter/" class="external">weather time series dataset</a> recorded by the <a href="https://www.bgc-jena.mpg.de" class="external">Max Planck Institute for Biogeochemistry</a>.
# 
# This dataset contains 14 different features such as air temperature, atmospheric pressure, and humidity. These were collected every 10 minutes, between 2009 and 2016. 

# # Time series forecasting

# In this study we will build a few different styles of models including Convolutional and Recurrent Neural Networks (CNNs and RNNs). The target feature is temperature
# 
# * Forecast for a single time step using single time steps as input
# * Forecast for a single time step using multiple time steps as input feature.
# * Forecast multiple steps:
#   * Single-shot: Make the predictions all at once.

# # Load the libraries:
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import plotly.express as px
import streamlit as st
import base64

def pd_prof(df):
  profile = ProfileReport(df_climate_hour, title="EDA Jena Climate Dataset")
  return profile

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title='Jena Climate Dataset',page_icon ="⛅", layout="wide")

file_ = open('Climate.gif', "rb")
contents = file_.read()
gif_data = base64.b64encode(contents).decode("utf-8")
file_.close()

# # # Load the Jena Climate Dataset
# df_climate = pd.read_csv('jena_climate_dataset_2009_2016.csv')
with st.container():
    st.title("Explore Jena Climate 🌇 Dataset and Compare DL Models") 
    st.markdown(
     f'<img src="data:image/gif;base64,{gif_data}" alt="Jena Climate Dataset" width="600">',
     unsafe_allow_html=True,)

st.subheader(''' Using this application the user can run Exploratory Data Analysis on the Jena Climate Dataset and also Compare the performance of ANN,CNN and RNN Time series models when predicting the Temperature.''')

st.write('---')
st.header('Jena Climate Dataset Selection 👇:')
file = st.radio("Choose Jena Climate Dataset",('Upload the Dataset','Use the Default Dataset'),label_visibility='collapsed')
if file == 'Upload the Dataset':
  uploaded_file = st.file_uploader(f"Select the Jena Climate Datset")
else:
  uploaded_file = 'jena_climate_dataset_2009_2016.csv'
if uploaded_file is not None:
  df_climate = pd.read_csv(uploaded_file)
  # In this study we will just deal with **hourly predictions**, so start by sub-sampling the data from 10-minute intervals to one-hour intervals:
# Slice [start:stop:step], starting from index 5 take every 6th record.
  df_climate_hour = df_climate[5::6]
  date_time = pd.to_datetime(df_climate_hour.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')

  plot_cols = ['T (degC)']
# profile = ProfileReport(df_climate_hour, title="EDA Jena Climate Dataset")
# st_profile_report(profile)

  st.write('---')
  st.subheader('Select the Mode of Execution')
  mode_main = st.radio('text', ("Exploratory Data Analysis",'Compare Deep Learning Models'),horizontal=True,label_visibility="collapsed")
  st.write('---')
  st.subheader(mode_main)
  if mode_main != 'Exploratory Data Analysis':
    st.write('---')
    st.markdown('##### Select the Deep Learning Models 👇:')
    dl_list = ['Linear Neural Network','Dense Neural Network','1D Convolutional Neural Network','LSTM Recurrent Neural Network', 'GRU Recurrent Neural Network']
    dl = st.multiselect("dl",options=dl_list,default = dl_list ,label_visibility='collapsed')
    st.markdown('##### Select the Prediction Method 👇:')
    option_window = st.selectbox('Mode',('Predict next hour temperature given previous hour data', 'Predict next hour temperature given previous 6 hours data', 'Predict next 24 hour temperature given previous 24 hours data',),label_visibility='collapsed')
    st.markdown('##### Select the Number of features 👇:')
    option = st.selectbox('Mode',('All Features','Time of the Day signal only'),label_visibility='collapsed')
    with st.expander('Information About the Deep Learning Models ℹ️'):
      st.write('Linear Neural Network - A tf.keras.layers.Dense layer with no activation set is a linear model.')
      st.write('Dense Neural Network - A tf.keras.layers.Dense layer with activation set is a linear model') 
      st.write('1D Convolutional Neural Network - A 1-D convolutional layer applies sliding convolutional filters to 1-D input')
      st.write('LSTM Recurrent Neural Network - Long Short-Term Memory (LSTM) networks are a type of recurrent neural network capable of learning order dependence in sequence prediction problems.')
      st.write('GRU Recurrent Neural Network - Gated Recurrent Unit(GRU) are networks that consist of gates which modulate the current input and the previous hidden state.') 
    st.write('---')
    download = st.radio('Do you want to Download the Best Performing Deep Learning Model?', ("Yes",'No'),horizontal=True)
    if st.button("Build and Compare Deep Learning Models"):
      wv = df_climate_hour['wv (m/s)']
      err_wv = wv == -9999.0
      wv[err_wv] = 0.0

      max_wv = df_climate_hour['max. wv (m/s)']
      err_max_wv = max_wv == -9999.0
      max_wv[err_max_wv] = 0.0

      # ### Feature Engineering
      # Converting the wind velocity columns into wind vector

      wv = df_climate_hour.pop('wv (m/s)')
      max_wv = df_climate_hour.pop('max. wv (m/s)')

      # Convert to radians.
      wd_rad = df_climate_hour.pop('wd (deg)')*np.pi / 180

      # Calculate the wind x and y components.
      df_climate_hour['Wx'] = wv*np.cos(wd_rad)
      df_climate_hour['Wy'] = wv*np.sin(wd_rad)

      # Calculate the max wind x and y components.
      df_climate_hour['max Wx'] = max_wv*np.cos(wd_rad)
      df_climate_hour['max Wy'] = max_wv*np.sin(wd_rad)

      # #### Time

      # `Date Time` column is very useful, but not in this string form. Start by converting it to seconds:
      timestamp_s = date_time.map(pd.Timestamp.timestamp)
      # Time in seconds is not a useful model input. Being weather data, it has clear daily and yearly periodicity. 
      # Get usable signals by using sine and cosine transforms to clear "Time of day" and "Time of year" signals:

      day = 24*60*60 # number of seconds in a day
      year = (365.2425)*day # number of seconds in a year

      df_climate_hour['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
      df_climate_hour['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
      df_climate_hour['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
      df_climate_hour['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

      if option != 'All Features':
        # ## Make Temperature predictions with only Time of day signal
        df = df_climate_hour[['T (degC)','Day sin','Day cos','Year sin','Year cos']]
      else:
        df = df_climate_hour.copy()

      
      # ### Train-Validation-Test Dataset split 

      # Using a `(70%, 20%, 10%)` split for the training, validation, and test sets.

      column_indices = {name: i for i, name in enumerate(df.columns)}

      n = len(df)
      train_df = df[0:int(n*0.7)]
      val_df = df[int(n*0.7):int(n*0.9)]
      test_df = df[int(n*0.9):]

      num_features = df.shape[1]


      Xaxis = ['Training', 'Validation', 'Test']
      Yaxis = [len(train_df), len(val_df), len(test_df)]
      plt.bar(Xaxis, Yaxis)
      # Displaying the bar plot

      # ### Normalize the data
      # 
      # It is important to scale features before training a neural network. Normalization is a common way of doing this scaling: subtract the mean and divide by the standard deviation of each feature.
      # The mean and standard deviation should only be computed using the training data so that the models have no access to the values in the validation and test sets.
      with st.spinner('Generating Distribution Plot for Selected Features'):
        train_mean = train_df.mean()
        train_std = train_df.std()

        train_df = (train_df - train_mean) / train_std
        val_df = (val_df - train_mean) / train_std
        test_df = (test_df - train_mean) / train_std


        # Now, peek at the distribution of the features. Some features do have long tails, but there are no obvious errors like the `-9999` wind velocity value.
        df_std = (df - train_mean) / train_std
        df_std = df_std.melt(var_name='Column', value_name='Normalized')
        fig = px.violin(df_std, x='Column', y='Normalized')
        st.write('---')
        st.subheader('Distribution Plot of Selected Features (Normalized)')
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
        st.write('---')
      
      class WindowGenerator():
          def __init__(self, input_width, label_width, shift,
                      train_df=train_df, val_df=val_df, test_df=test_df,
                      label_columns=None):
            # Store the raw data.
            self.train_df = train_df
            self.val_df = val_df
            self.test_df = test_df

            # Work out the label column indices.
            self.label_columns = label_columns
            if label_columns is not None:
              self.label_columns_indices = {name: i for i, name in
                                            enumerate(label_columns)}
            self.column_indices = {name: i for i, name in
                                  enumerate(train_df.columns)}

            # Work out the window parameters.
            self.input_width = input_width
            self.label_width = label_width
            self.shift = shift

            self.total_window_size = input_width + shift

            self.input_slice = slice(0, input_width)
            self.input_indices = np.arange(self.total_window_size)[self.input_slice]

            self.label_start = self.total_window_size - self.label_width
            self.labels_slice = slice(self.label_start, None)
            self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

          def __repr__(self):
            return '\n'.join([
                f'Total window size: {self.total_window_size}',
                f'Input indices: {self.input_indices}',
                f'Label indices: {self.label_indices}',
                f'Label column name(s): {self.label_columns}'])


      w1 = WindowGenerator(input_width=6, label_width=1, shift=1,
                            label_columns=['T (degC)'])

      w2 = WindowGenerator(input_width=24, label_width=1, shift=24,
                            label_columns=['T (degC)'])

        # ### 2. Split
        # 
        # Given a list of consecutive inputs, the `split_window` method will convert them to a window of inputs and a window of labels.
        # 
        # w2 will be split as shown below the window size will be 7, 6 inputs each of 19 features and 1 output

        # ![image_2023-01-05_144926267.png](attachment:image_2023-01-05_144926267.png)


      def split_window(self, features):
          inputs = features[:, self.input_slice, :]
          labels = features[:, self.labels_slice, :]
          if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)

          # Slicing doesn't preserve static shape information, so set the shapes
          # manually. This way the `tf.data.Datasets` are easier to inspect.
          inputs.set_shape([None, self.input_width, None])
          labels.set_shape([None, self.label_width, None])

          return inputs, labels

      WindowGenerator.split_window = split_window


        # Stack three slices, the length of the total window.
      example_window = tf.stack([np.array(train_df[:w1.total_window_size]),
                                  np.array(train_df[100:100+w1.total_window_size]),
                                  np.array(train_df[200:200+w1.total_window_size])])

      example_inputs, example_labels = w1.split_window(example_window)

      
      w1.example = example_inputs, example_labels

      def plot(self, model=None, plot_col='T (degC)', max_subplots=3):
          inputs, labels = self.example
          plt.figure(figsize=(10, 6))
          plot_col_index = self.column_indices[plot_col]
          max_n = min(max_subplots, len(inputs))
          for n in range(max_n):
            plt.subplot(max_n, 1, n+1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                    label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
              label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
              label_col_index = plot_col_index

            if label_col_index is None:
              continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
              predictions = model(inputs)
              plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                          marker='X', edgecolors='k', label='Predictions',
                          c='#ff7f0e', s=64)

            if n == 0:
              plt.legend()

          plt.xlabel('Time [h]')
          return plt
      WindowGenerator.plot = plot

      w1.example = example_inputs, example_labels


        # ### 4. Create `tf.data.Dataset`s

        # Finally, this `make_dataset` method will take a time series DataFrame and convert it to a `tf.data.Dataset` of `(input_window, label_window)` pairs using the `tf.keras.utils.timeseries_dataset_from_array` function:

      def make_dataset(self, data):
          data = np.array(data, dtype=np.float32)
          ds = tf.keras.utils.timeseries_dataset_from_array(
              data=data,
              targets=None,
              sequence_length=self.total_window_size,
              sequence_stride=1,
              shuffle=True,
              batch_size=32,)

          ds = ds.map(self.split_window)

          return ds

      WindowGenerator.make_dataset = make_dataset


      @property
      def train(self):
          return self.make_dataset(self.train_df)

      @property
      def val(self):
          return self.make_dataset(self.val_df)

      @property
      def test(self):
          return self.make_dataset(self.test_df)

      @property
      def example(self):
          """Get and cache an example batch of `inputs, labels` for plotting."""
          result = getattr(self, '_example', None)
          if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
          return result

      WindowGenerator.train = train
      WindowGenerator.val = val
      WindowGenerator.test = test
      WindowGenerator.example = example

      def compile_and_fit(model, window, patience=2):
          early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                            patience=patience,
                                                            mode='min')

          model.compile(loss=tf.keras.losses.MeanSquaredError(),
                        optimizer=tf.keras.optimizers.Adam(),
                        metrics=[tf.keras.metrics.MeanAbsoluteError()])

          history = model.fit(window.train, epochs=MAX_EPOCHS,
                              validation_data=window.val,
                              callbacks=[early_stopping])
          return history
      if option_window == 'Predict next hour temperature given previous hour data':
        single_step_window = WindowGenerator(
                  input_width=1, label_width=1, shift=1,
                  label_columns=['T (degC)'])

       ## Building Deep Learning Models

        MAX_EPOCHS = 20
        val_performance = {}
        performance = {}
        wide_window = WindowGenerator(
                    input_width=24, label_width=24, shift=1,
                    label_columns=['T (degC)'])



          # dl_list = ['Linear Neural Network','Dense Neural Network','1D Convolutional Neural Network','LSTM Recurrent Neural Network', 'GRU Recurrent Neural Network']
        if 'Linear Neural Network' in dl:
            # ## Approach 1 - Forecast for a single time step using single time steps as input
            # ## Single step models
            # The simplest model you can build on this sort of data is one that predicts a single feature's value—1 time step (one hour) into the future based only on the current conditions.
            # building models to predict the `T (degC)` value one hour into the future.

            with st.spinner('Building Linear Neural Network Model'):
              st.subheader('Linear Neural Network Model Performance')
              
                # ### Linear model

                # A `tf.keras.layers.Dense` layer with no `activation` set is a linear model. The layer only transforms the last axis of the data from `(batch, time, inputs)` to `(batch, time, units)`; it is applied independently to every item across the `batch` and `time` axes.

              linear = tf.keras.Sequential([
                    tf.keras.layers.Dense(units=1)
                ])


              history = compile_and_fit(linear, single_step_window)
              val_performance['Linear'] = linear.evaluate(single_step_window.val)
              performance['Linear'] = linear.evaluate(single_step_window.test, verbose=0)
                
              plot = wide_window.plot(linear)    
              st.pyplot(plot)
              linear_model = history
              model = linear
        if 'Dense Neural Network' in dl:
          with st.spinner('Building Dense Neural Network Model'):
            st.subheader('Dense Neural Network Model Performance')
            dense = tf.keras.Sequential([
            tf.keras.layers.Dense(units=64, activation='relu'),
            tf.keras.layers.Dense(units=64, activation='relu'),
            tf.keras.layers.Dense(units=1)
            ])

            history = compile_and_fit(dense, single_step_window)

            val_performance['Dense'] = dense.evaluate(single_step_window.val)
            performance['Dense'] = dense.evaluate(single_step_window.test, verbose=0)

            plot_dense = wide_window.plot(dense)
            st.pyplot(plot_dense)
            dense_model = history
            model = dense
        if '1D Convolutional Neural Network' in dl:
          with st.spinner('Building 1D Convolutional Neural Network Model'):
            st.subheader('1D Convolutional Neural Network Model Performance')
            CONV_WIDTH = 1
            conv_window = WindowGenerator(
              input_width=CONV_WIDTH,
              label_width=1,
              shift=1,
              label_columns=['T (degC)'])


            conv_model = tf.keras.Sequential([
                tf.keras.layers.Conv1D(filters=32,
                                      kernel_size=(CONV_WIDTH,),
                                      activation='relu'),
                tf.keras.layers.Dense(units=32, activation='relu'),
                tf.keras.layers.Dense(units=1),
            ])
            history = compile_and_fit(conv_model, conv_window)

            val_performance['Conv'] = conv_model.evaluate(conv_window.val)
            performance['Conv'] = conv_model.evaluate(conv_window.test, verbose=0)
            LABEL_WIDTH = 24
            INPUT_WIDTH = LABEL_WIDTH + (CONV_WIDTH - 1)
            wide_conv_window = WindowGenerator(
                input_width=INPUT_WIDTH,
                label_width=LABEL_WIDTH,
                shift=1,
                label_columns=['T (degC)'])

            plot = wide_conv_window.plot(conv_model)
            st.pyplot(plot)
            conv = history
            model = conv_model
        if 'LSTM Recurrent Neural Network' in dl:
          with st.spinner('Building LSTM Recurrent Neural Network Model'):
            st.subheader('LSTM Recurrent Neural Network Model Performance')
            lstm_model = tf.keras.models.Sequential([
            # Shape [batch, time, features] => [batch, time, lstm_units]
            tf.keras.layers.LSTM(32, return_sequences=True),
            # Shape => [batch, time, features]
            tf.keras.layers.Dense(units=1)
            ])

            history = compile_and_fit(lstm_model, wide_window)
            val_performance['LSTM'] = lstm_model.evaluate(wide_window.val)
            performance['LSTM'] = lstm_model.evaluate(wide_window.test, verbose=0)
            plot = wide_window.plot(lstm_model)
            st.pyplot(plot)
            lstm = history
            model = lstm_model
        if 'GRU Recurrent Neural Network' in dl:
          with st.spinner('Building GRU Recurrent Neural Network Model'):
            st.subheader('GRU Recurrent Neural Network Model Performance')
            gru_model = tf.keras.models.Sequential([
                # Shape [batch, time, features] => [batch, time, lstm_units]
                tf.keras.layers.GRU(32, return_sequences=True),
                # Shape => [batch, time, features]
                tf.keras.layers.Dense(units=1)
            ])
            history = compile_and_fit(gru_model, wide_window)
            val_performance['GRU'] = gru_model.evaluate(wide_window.val)
            performance['GRU'] = gru_model.evaluate(wide_window.test, verbose=0)
            plot = wide_window.plot(gru_model)
            st.pyplot(plot)
            gru =history
            model = gru_model
        if len(val_performance.values()) != 0: 
          st.write('---')
          st.subheader('Performance Comparision of Deep Learning Models') 
          tab1,tab2 = st.tabs(["Performance Comparision Graph 📊","Performance Metrics 🔢"])
          with tab1:
            x = np.arange(len(performance))
            width = 0.3
            metric_name = 'mean_absolute_error'
            metric_index = model.metrics_names.index('mean_absolute_error')
            val_mae = [v[metric_index] for v in val_performance.values()]
            test_mae = [v[metric_index] for v in performance.values()]

            plt.clf()
            plt.ylabel('mean_absolute_error [T (degC), normalized]')
            plt.bar(x - 0.17, val_mae, width, label='Validation')
            plt.bar(x + 0.17, test_mae, width, label='Test')
            plt.xticks(ticks=x, labels=performance.keys(),
                      rotation=45)
            _ = plt.legend()
            st.pyplot(plt)
          with tab2:  
            perform = {}
            for name, value in performance.items():
              perform[name] = value[1]
            perform = sorted(perform.items(), key=lambda x:x[1], reverse=False)
            perform_df = pd.DataFrame(perform)
            perform_df.columns = ['Deep Learning Model',"MAE Score"]
            st.write(perform_df)
          st.write('---')
          st.subheader('Performance Summary')
          st.info(f"The {perform[0][0]} Neural Network is the best performing model with MAE value of {perform[0][1]:0.4f}")
          st.write('---')
          if download == 'Yes':
            if perform[0][0] == 'Linear':
              linear.save(f'Jena_{perform[0][0]}_1hour_in_1hour_out.h5')
            elif perform[0][0] == 'Dense':  
              dense.save(f'Jena_{perform[0][0]}_1hour_in_1hour_out.h5')
            elif perform[0][0] == 'Conv':
              conv_model.save(f'Jena_{perform[0][0]}_1hour_in_1hour_out.h5')
            elif perform[0][0] == 'LSTM':
              lstm_model.save(f'Jena_{perform[0][0]}_1hour_in_1hour_out.h5')
            else:
              gru_model.save(f'Jena_{perform[0][0]}_1hour_in_1hour_out.h5')
            st.write('---')  
            st.success(f"Jena_{perform[0][0]}_1hour_in_1hour_out.h5 downloaded successfully")
      elif option_window == 'Predict next hour temperature given previous 6 hours data':
        MAX_EPOCHS = 20
        val_performance = {}
        performance = {}
        CONV_WIDTH = 6
        multi_window = WindowGenerator(
            input_width=CONV_WIDTH,
            label_width=1,
            shift=1,
            label_columns=['T (degC)'])
        if 'Linear Neural Network' in dl:
          with st.spinner('Building Linear Neural Network Model'):
            st.subheader('Linear Neural Network Model Performance')
            linear = tf.keras.Sequential([
                tf.keras.layers.Dense(units=1),
                tf.keras.layers.Reshape([1, -1])
            ])
    
            history = compile_and_fit(linear, multi_window)

            val_performance['Linear'] = linear.evaluate(multi_window.val)
            performance['Linear'] = linear.evaluate(multi_window.test, verbose=0)
            plot = multi_window.plot(linear)    
            st.pyplot(plot)
            linear_model = history
            model = linear
        if 'Dense Neural Network' in dl:
          with st.spinner('Building Dense Neural Network Model'):
            st.subheader('Dense Neural Network Model Performance')
            CONV_WIDTH = 6
            conv_window = WindowGenerator(
            input_width=CONV_WIDTH,
            label_width=1,
            shift=1,
            label_columns=['T (degC)'])

            dense = tf.keras.Sequential([
            # Shape: (time, features) => (time*features)
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Dense(units=1),
            # Add back the time dimension.
            # Shape: (outputs) => (1, outputs)
            tf.keras.layers.Reshape([1, -1]),
        ])

            history = compile_and_fit(dense, conv_window)

            val_performance['Dense'] = dense.evaluate(conv_window.val)
            performance['Dense'] = dense.evaluate(conv_window.test, verbose=0)
            plot_dense = conv_window.plot(dense)
            st.pyplot(plot_dense)
            dense_model = history
            model = dense


        if '1D Convolutional Neural Network' in dl:
          with st.spinner('Building 1D Convolutional Neural Network Model'):
            st.subheader('1D Convolutional Neural Network Model Performance')
            CONV_WIDTH = 6
            conv_window = WindowGenerator(
            input_width=CONV_WIDTH,
            label_width=1,
            shift=1,
            label_columns=['T (degC)'])

            conv_model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(filters=32,
                                  kernel_size=(CONV_WIDTH,),
                                  activation='relu'),
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Dense(units=1),
            ])

            history = compile_and_fit(conv_model, conv_window)
            val_performance['Conv'] = conv_model.evaluate(conv_window.val)
            performance['Conv'] = conv_model.evaluate(conv_window.test, verbose=0)

            LABEL_WIDTH = 24
            INPUT_WIDTH = LABEL_WIDTH + (CONV_WIDTH - 1)
            wide_conv_window = WindowGenerator(
                input_width=INPUT_WIDTH,
                label_width=LABEL_WIDTH,
                shift=1,
                label_columns=['T (degC)'])            
            plot = wide_conv_window.plot(conv_model)
            st.pyplot(plot)
            conv = history
            model = conv_model
        if 'LSTM Recurrent Neural Network' in dl:
          with st.spinner('Building LSTM Recurrent Neural Network Model'):
            st.subheader('LSTM Recurrent Neural Network Model Performance')
            num_features = df.shape[1]
            wide_window = WindowGenerator(input_width=6,
                               label_width=1,
                               shift=1)
            lstm_model = tf.keras.Sequential([
            # Shape [batch, time, features] => [batch, lstm_units].
            # Adding more `lstm_units` just overfits more quickly.
            tf.keras.layers.LSTM(32, return_sequences=False),
            # Shape => [batch, out_steps*features].
            tf.keras.layers.Dense(num_features,
                                  kernel_initializer=tf.initializers.zeros()),
            # Shape => [batch, out_steps, features].
            tf.keras.layers.Reshape([1, num_features])
        ])

            history = compile_and_fit(lstm_model, wide_window)
            val_performance['LSTM'] = lstm_model.evaluate(wide_window.val)
            performance['LSTM'] = lstm_model.evaluate(wide_window.test, verbose=0)
            plot = wide_window.plot(lstm_model)
            st.pyplot(plot)
            lstm = history
            model = lstm_model
        if 'GRU Recurrent Neural Network' in dl:
          with st.spinner('Building GRU Recurrent Neural Network Model'):
            st.subheader('GRU Recurrent Neural Network Model Performance')
            wide_window = WindowGenerator(input_width=6,
                               label_width=1,
                               shift=1)
            gru_model = tf.keras.Sequential([
            # Shape [batch, time, features] => [batch, lstm_units].
            # Adding more `lstm_units` just overfits more quickly.
            tf.keras.layers.GRU(32, return_sequences=False),
            # Shape => [batch, out_steps*features].
            tf.keras.layers.Dense(num_features,
                                  kernel_initializer=tf.initializers.zeros()),
            # Shape => [batch, out_steps, features].
            tf.keras.layers.Reshape([1, num_features])
        ])
            history = compile_and_fit(gru_model, wide_window)
            val_performance['GRU'] = gru_model.evaluate(wide_window.val)
            performance['GRU'] = gru_model.evaluate(wide_window.test, verbose=0)
            plot = wide_window.plot(gru_model)
            st.pyplot(plot)
            gru =history
            model = gru_model
        if len(val_performance.values()) != 0:
          st.write('---')
          st.subheader('Performance Comparision of Deep Learning Models') 
          tab1,tab2 = st.tabs(["Performance Comparision Graph 📊","Performance Metrics 🔢"])
          with tab1:
            x = np.arange(len(performance))
            width = 0.3
            metric_name = 'mean_absolute_error'
            metric_index = model.metrics_names.index('mean_absolute_error')
            val_mae = [v[metric_index] for v in val_performance.values()]
            test_mae = [v[metric_index] for v in performance.values()]

            plt.clf()
            plt.ylabel('mean_absolute_error [T (degC), normalized]')
            plt.bar(x - 0.17, val_mae, width, label='Validation')
            plt.bar(x + 0.17, test_mae, width, label='Test')
            plt.xticks(ticks=x, labels=performance.keys(),
                      rotation=45)
            _ = plt.legend()
            st.pyplot(plt)
          with tab2:  
            perform = {}
            for name, value in performance.items():
              perform[name] = value[1]
            perform = sorted(perform.items(), key=lambda x:x[1], reverse=False)
            perform_df = pd.DataFrame(perform)
            perform_df.columns = ['Deep Learning Model',"MAE Score"]
            st.write(perform_df)
          st.write('---')
          st.subheader('Performance Summary')
          st.info(f"The {perform[0][0]} Neural Network is the best performing model with MAE value of {perform[0][1]:0.4f}")
          st.write('---')
          if download == 'Yes':
            if perform[0][0] == 'Linear':
              linear.save(f'Jena_{perform[0][0]}_6hour_in_1hour_out.h5')
            elif perform[0][0] == 'Dense':  
              dense.save(f'Jena_{perform[0][0]}_6hour_in_1hour_out.h5')
            elif perform[0][0] == 'Conv':
              conv_model.save(f'Jena_{perform[0][0]}_6hour_in_1hour_out.h5')
            elif perform[0][0] == 'LSTM':
              lstm_model.save(f'Jena_{perform[0][0]}_6hour_in_1hour_out.h5')
            else:
              gru_model.save(f'Jena_{perform[0][0]}_6hour_in_1hour_out.h5')
            st.write('---')  
            st.success(f"Jena_{perform[0][0]}_6hour_in_1hour_out.h5 downloaded successfully")  
      else:
        OUT_STEPS = 24        
        num_features = df.shape[1]
        multi_window = WindowGenerator(input_width=24,
                               label_width=OUT_STEPS,
                               shift=OUT_STEPS)
        multi_val_performance = {}
        multi_performance = {}
        MAX_EPOCHS = 20          
        # ### Linear Model
        if 'Linear Neural Network' in dl:
          with st.spinner('Building Linear Neural Network Model'):
            st.subheader('Linear Neural Network Model Performance')
            multi_linear_model = tf.keras.Sequential([
                # Take the last time-step.
                # Shape [batch, time, features] => [batch, 1, features]
                tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
                # Shape => [batch, 1, out_steps*features]
                tf.keras.layers.Dense(OUT_STEPS*num_features,
                                      kernel_initializer=tf.initializers.zeros()),
                # Shape => [batch, out_steps, features]
                tf.keras.layers.Reshape([OUT_STEPS, num_features])
            ])

            history = compile_and_fit(multi_linear_model, multi_window)
            multi_val_performance['Linear'] = multi_linear_model.evaluate(multi_window.val)
            multi_performance['Linear'] = multi_linear_model.evaluate(multi_window.test, verbose=0)
            plot = multi_window.plot(multi_linear_model)    
            st.pyplot(plot)
            model = multi_linear_model
        if 'Dense Neural Network' in dl:
          with st.spinner('Building Dense Neural Network Model'):
            st.subheader('Dense Neural Network Model Performance')
            
            # #### Dense
            multi_dense_model = tf.keras.Sequential([
                # Take the last time step.
                # Shape [batch, time, features] => [batch, 1, features]
                tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
                # Shape => [batch, 1, dense_units]
                tf.keras.layers.Dense(512, activation='relu'),
                # Shape => [batch, out_steps*features]
                tf.keras.layers.Dense(OUT_STEPS*num_features,
                                      kernel_initializer=tf.initializers.zeros()),
                # Shape => [batch, out_steps, features]
                tf.keras.layers.Reshape([OUT_STEPS, num_features])
            ])

            history = compile_and_fit(multi_dense_model, multi_window)

            multi_val_performance['Dense'] = multi_dense_model.evaluate(multi_window.val)
            multi_performance['Dense'] = multi_dense_model.evaluate(multi_window.test, verbose=0)
            plot = multi_window.plot(multi_dense_model)    
            st.pyplot(plot)
            model = multi_dense_model
        if '1D Convolutional Neural Network' in dl:
          with st.spinner('Building 1D Convolutional Neural Network Model'):
            st.subheader('1D Convolutional Neural Network Model Performance')
            CONV_WIDTH = 3
            multi_conv_model = tf.keras.Sequential([
                # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
                tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
                # Shape => [batch, 1, conv_units]
                tf.keras.layers.Conv1D(256, activation='relu', kernel_size=(CONV_WIDTH)),
                # Shape => [batch, 1,  out_steps*features]
                tf.keras.layers.Dense(OUT_STEPS*num_features,
                                      kernel_initializer=tf.initializers.zeros()),
                # Shape => [batch, out_steps, features]
                tf.keras.layers.Reshape([OUT_STEPS, num_features])
            ])

            history = compile_and_fit(multi_conv_model, multi_window)
            multi_val_performance['Conv'] = multi_conv_model.evaluate(multi_window.val)
            multi_performance['Conv'] = multi_conv_model.evaluate(multi_window.test, verbose=0)
            plot = multi_window.plot(multi_conv_model)
            st.pyplot(plot)
            conv = history
            model = multi_conv_model
        if 'LSTM Recurrent Neural Network' in dl:
          with st.spinner('Building LSTM Recurrent Neural Network Model'):
            st.subheader('LSTM Recurrent Neural Network Model Performance')
            
            multi_lstm_model = tf.keras.Sequential([
                # Shape [batch, time, features] => [batch, lstm_units].
                # Adding more `lstm_units` just overfits more quickly.
                tf.keras.layers.LSTM(32, return_sequences=False),
                # Shape => [batch, out_steps*features].
                tf.keras.layers.Dense(OUT_STEPS*num_features,
                                      kernel_initializer=tf.initializers.zeros()),
                # Shape => [batch, out_steps, features].
                tf.keras.layers.Reshape([OUT_STEPS, num_features])
            ])

            history = compile_and_fit(multi_lstm_model, multi_window)

            multi_val_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.val)
            multi_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.test, verbose=0)
            plot = multi_window.plot(multi_lstm_model)
            st.pyplot(plot)
            conv = history
            model = multi_lstm_model
        if 'GRU Recurrent Neural Network' in dl:
          with st.spinner('Building GRU Recurrent Neural Network Model'):
            st.subheader('GRU Recurrent Neural Network Model Performance')
            multi_gru_model = tf.keras.Sequential([
            # Shape [batch, time, features] => [batch, lstm_units].
            # Adding more `lstm_units` just overfits more quickly.
            tf.keras.layers.GRU(32, return_sequences=False),
            # Shape => [batch, out_steps*features].
            tf.keras.layers.Dense(OUT_STEPS*num_features,
                                  kernel_initializer=tf.initializers.zeros()),
            # Shape => [batch, out_steps, features].
            tf.keras.layers.Reshape([OUT_STEPS, num_features])
            ])
            history = compile_and_fit(multi_gru_model, multi_window)
            multi_val_performance['GRU'] = multi_gru_model.evaluate(multi_window.val)
            multi_performance['GRU'] = multi_gru_model.evaluate(multi_window.test, verbose=0)            
            plot = multi_window.plot(multi_gru_model)
            st.pyplot(plot)
            conv = history
            model = multi_gru_model
        if len(multi_val_performance.values()) != 0:
          st.write('---')
          st.subheader('Performance Comparision of Deep Learning Models') 
          tab1,tab2 = st.tabs(["Performance Comparision Graph 📊","Performance Metrics 🔢"])
          with tab1:
            x = np.arange(len(multi_performance))
            width = 0.3

            metric_name = 'mean_absolute_error'
            metric_index = model.metrics_names.index('mean_absolute_error')
            val_mae = [v[metric_index] for v in multi_val_performance.values()]
            test_mae = [v[metric_index] for v in multi_performance.values()]
            plt.clf()
            plt.bar(x - 0.17, val_mae, width, label='Validation')
            plt.bar(x + 0.17, test_mae, width, label='Test')
            plt.xticks(ticks=x, labels=multi_performance.keys(),
                      rotation=45)
            plt.ylabel(f'MAE (average over all times and outputs)')
            _ = plt.legend()
            st.pyplot(plt)
          with tab2:  
            perform = {}
            for name, value in multi_performance.items():
              perform[name] = value[1]
            perform = sorted(perform.items(), key=lambda x:x[1], reverse=False)
            perform_df = pd.DataFrame(perform)
            perform_df.columns = ['Deep Learning Model',"MAE Score"]
            st.write(perform_df)
          st.write('---')
          st.subheader('Performance Summary')
          st.info(f"The {perform[0][0]} Neural Network is the best performing model with MAE value of {perform[0][1]:0.4f}")
          st.write('---')
          if download == 'Yes':
            if perform[0][0] == 'Linear':
              multi_linear_model.save(f'Jena_{perform[0][0]}_24hour_in_24hour_out.h5')
            elif perform[0][0] == 'Dense':  
              multi_dense_model.save(f'Jena_{perform[0][0]}_24hour_in_24hour_out.h5')
            elif perform[0][0] == 'Conv':
              multi_conv_model.save(f'Jena_{perform[0][0]}_24hour_in_24hour_out.h5')
            elif perform[0][0] == 'LSTM':
              multi_lstm_model.save(f'Jena_{perform[0][0]}_24hour_in_24hour_out.h5')
            else:
              multi_gru_model.save(f'Jena_{perform[0][0]}_24hour_in_24hour_out.h5')  
            st.write('---')  
            st.success(f"Jena_{perform[0][0]}_24hour_in_24hour_out.h5 downloaded successfully")
  else: 
    tab1,tab2,tab3,tab4 = st.tabs(["File 📄 Preview🔎","Temperature T (degC) Yearly 🗓️ Distribution📈","Temperature T (degC) Daily 📅 Distribution📈","Temperature T (degC) Hourly 🕐 Distribution📈"])
    with tab1:
      st.subheader('Selected File 📄 Preview🔎 sub-sampling the data from 10-minute intervals to one-hour intervals:')
      st.dataframe(df_climate_hour)
    with tab2:
      st.subheader("Temperature T (degC) Yearly 🗓️ Distribution📈")
      plot_features = df_climate_hour[plot_cols]
      plot_features.index = date_time
      fig = px.line(plot_features,x=plot_features.index, y='T (degC)')
      st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    with tab3:  
      st.subheader("Temperature T (degC) Daily 📅 Distribution📈")
      plot_features = df_climate_hour[plot_cols][:480]
      plot_features.index = date_time[:480]
      fig = px.line(plot_features,x=plot_features.index, y='T (degC)')
      st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    with tab4:
      st.subheader("Temperature T (degC) Hourly 🕐 Distribution📈")
      plot_features = df_climate_hour[plot_cols][:24]
      plot_features.index = date_time[:24]
      fig = px.line(plot_features,x=plot_features.index, y='T (degC)')
      st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    if st.button("Generate Pandas Profiling Report"):
      st.write('---')
      st.subheader("Pandas Profiling Report 📝")
      profile = pd_prof(df_climate_hour)
      st_profile_report(profile)
      export=profile.to_html()
      st.download_button(label="Download Full Report", data=export, file_name='Jena EDA report.html')
