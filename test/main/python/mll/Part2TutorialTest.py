import unittest

import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False


class Part2TutorialTest(unittest.TestCase):
    def test_whole_example(self):
        # # When run everything
        # np.testing.assert_allclose(multi_step_history.history['loss']
        #             , [0.4961, 0.3468, 0.3278, 0.2430, 0.1988, 0.2062, 0.1973, 0.1958, 0.1965, 0.1884], atol=0.01)
        # np.testing.assert_allclose(multi_step_history.history['val_loss']
        #             , [0.2883, 0.2905, 0.2478, 0.2091, 0.2135, 0.2065, 0.2070, 0.1916, 0.1855, 0.1960], atol=0.01)
        #
        # # When run just whole Phase 2
        # np.testing.assert_allclose(multi_step_history.history['loss']
        #             , [0.4961, 0.3468, 0.3278, 0.2430, 0.1988, 0.2062, 0.1973, 0.1958, 0.1965, 0.1884], atol=0.063)
        # np.testing.assert_allclose(multi_step_history.history['val_loss']
        #             , [0.2883, 0.2905, 0.2478, 0.2091, 0.2135, 0.2065, 0.2070, 0.1916, 0.1855, 0.1960], atol=0.031)
        zip_path = tf.keras.utils.get_file(
            origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
            fname='jena_climate_2009_2016.csv.zip',
            extract=True)
        csv_path, _ = os.path.splitext(zip_path)
        df = pd.read_csv(csv_path)

        def univariate_data(dataset, start_index, end_index, history_size, target_size):
            data = []
            labels = []

            start_index = start_index + history_size
            if end_index is None:
                end_index = len(dataset) - target_size

            for i in range(start_index, end_index):
                indices = range(i - history_size, i)
                # Reshape data from (history_size,) to (history_size, 1)
                data.append(np.reshape(dataset[indices], (history_size, 1)))
                labels.append(dataset[i + target_size])
            return np.array(data), np.array(labels)

        TRAIN_SPLIT = 300000
        tf.random.set_seed(13)
        uni_data = df['T (degC)']
        uni_data.index = df['Date Time']
        uni_data.head()
        uni_data.plot(subplots=True)
        uni_data = uni_data.values
        uni_train_mean = uni_data[:TRAIN_SPLIT].mean()
        uni_train_std = uni_data[:TRAIN_SPLIT].std()
        uni_data = (uni_data - uni_train_mean) / uni_train_std
        univariate_past_history = 20
        univariate_future_target = 0

        x_train_uni, y_train_uni = univariate_data(uni_data, 0, TRAIN_SPLIT,
                                                   univariate_past_history,
                                                   univariate_future_target)
        x_val_uni, y_val_uni = univariate_data(uni_data, TRAIN_SPLIT, None,
                                               univariate_past_history,
                                               univariate_future_target)
        print('Single window of past history')
        print(x_train_uni[0])
        print('\n Target temperature to predict')
        print(y_train_uni[0])

        def create_time_steps(length):
            return list(range(-length, 0))

        def show_plot(plot_data, delta, title):
            labels = ['History', 'True Future', 'Model Prediction']
            marker = ['.-', 'rx', 'go']
            time_steps = create_time_steps(plot_data[0].shape[0])
            if delta:
                future = delta
            else:
                future = 0

            plt.title(title)
            for i, x in enumerate(plot_data):
                if i:
                    plt.plot(future, plot_data[i], marker[i], markersize=10,
                             label=labels[i])
                else:
                    plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
            plt.legend()
            plt.xlim([time_steps[0], (future + 5) * 2])
            plt.xlabel('Time-Step')
            return plt

        def baseline(history):
            return np.mean(history)

        BATCH_SIZE = 256
        BUFFER_SIZE = 10000

        train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
        train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

        val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
        val_univariate = val_univariate.batch(BATCH_SIZE).repeat()
        simple_lstm_model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(8, input_shape=x_train_uni.shape[-2:]),
            tf.keras.layers.Dense(1)
        ])

        simple_lstm_model.compile(optimizer='adam', loss='mae')

        for x, y in val_univariate.take(1):
            print(simple_lstm_model.predict(x).shape)

        EVALUATION_INTERVAL = 200
        EPOCHS = 10

        simple_lstm_model.fit(train_univariate, epochs=EPOCHS,
                              steps_per_epoch=EVALUATION_INTERVAL,
                              validation_data=val_univariate, validation_steps=50)
        np.testing.assert_allclose([simple_lstm_model.predict(x)[0][0] for x, _ in val_univariate.take(3)]
                                   , [0.58993083, 0.646739, 1.33638])

        ## Part 2: Forecast a multivariate time series
        features_considered = ['p (mbar)', 'T (degC)', 'rho (g/m**3)']
        features = df[features_considered]
        features.index = df['Date Time']

        dataset = features.values
        data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
        data_std = dataset[:TRAIN_SPLIT].std(axis=0)
        dataset = (dataset - data_mean) / data_std

        def multivariate_data(dataset, target, start_index, end_index, history_size,
                              target_size, step, single_step=False):
            data = []
            labels = []

            start_index = start_index + history_size
            if end_index is None:
                end_index = len(dataset) - target_size

            for i in range(start_index, end_index):
                indices = range(i - history_size, i, step)
                data.append(dataset[indices])

                if single_step:
                    labels.append(target[i + target_size])
                else:
                    labels.append(target[i:i + target_size])

            return np.array(data), np.array(labels)

        past_history = 720
        future_target = 72
        STEP = 6

        x_train_single, y_train_single = multivariate_data(dataset, dataset[:, 1], 0,
                                                           TRAIN_SPLIT, past_history,
                                                           future_target, STEP,
                                                           single_step=True)
        x_val_single, y_val_single = multivariate_data(dataset, dataset[:, 1],
                                                       TRAIN_SPLIT, None, past_history,
                                                       future_target, STEP,
                                                       single_step=True)
        print('Single window of past history : {}'.format(x_train_single[0].shape))
        train_data_single = tf.data.Dataset.from_tensor_slices((x_train_single, y_train_single))
        train_data_single = train_data_single.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

        val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
        val_data_single = val_data_single.batch(BATCH_SIZE).repeat()
        single_step_model = tf.keras.models.Sequential()
        single_step_model.add(tf.keras.layers.LSTM(32,
                                                   input_shape=x_train_single.shape[-2:]))
        single_step_model.add(tf.keras.layers.Dense(1))

        single_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')
        for x, y in val_data_single.take(1):
            print(single_step_model.predict(x).shape)

        single_step_history = single_step_model.fit(train_data_single, epochs=EPOCHS,
                                                    steps_per_epoch=EVALUATION_INTERVAL,
                                                    validation_data=val_data_single,
                                                    validation_steps=50)
        np.testing.assert_allclose(single_step_history.history['loss']
                , [0.3089671531319618,0.2623981712013483,0.26122886791825295,0.25684389054775236,0.22617250859737395,
                    0.24156176969993806,0.24054658599197865,0.23966932341456412,0.24450092546641827,0.23804798915982248]
                , atol=0.001)
        np.testing.assert_allclose(single_step_history.history['val_loss']
                , [0.2646562772989273, 0.2438293693959713, 0.2471142350882292, 0.24247292056679726, 0.2365048633515835
                , 0.2671605175733566, 0.25608043298125266, 0.23974552996456622, 0.24380892500281334, 0.2447321417927742]
                , atol=0.0045)
        np.testing.assert_allclose([single_step_model.predict(x)[0][0] for x, _ in val_data_single.take(3)]
                                   , [1.4829265, 0.7117876, 0.14071949])
        ## Multi-Step model
        future_target = 72
        x_train_multi, y_train_multi = multivariate_data(dataset, dataset[:, 1], 0,
                                                         TRAIN_SPLIT, past_history,
                                                         future_target, STEP)
        x_val_multi, y_val_multi = multivariate_data(dataset, dataset[:, 1],
                                                     TRAIN_SPLIT, None, past_history,
                                                     future_target, STEP)
        print('Single window of past history : {}'.format(x_train_multi[0].shape))
        print('\n Target temperature to predict : {}'.format(y_train_multi[0].shape))
        train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
        train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

        val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
        val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()

        def multi_step_plot(history, true_future, prediction):
            plt.figure(figsize=(12, 6))
            num_in = create_time_steps(len(history))
            num_out = len(true_future)

            plt.plot(num_in, np.array(history[:, 1]), label='History')
            plt.plot(np.arange(num_out) / STEP, np.array(true_future), 'bo',
                     label='True Future')
            if prediction.any():
                plt.plot(np.arange(num_out) / STEP, np.array(prediction), 'ro',
                         label='Predicted Future')
            plt.legend(loc='upper left')
            plt.show()

        for x, y in train_data_multi.take(1):
            np.testing.assert_allclose(y[0].numpy(), [-0.9648635, -0.90118192, -0.88497207, -0.86065728, -0.85834159,
                -0.86181513, -0.8537102, -0.82939542, -0.85023666, -0.90118192, -0.93707518, -0.94170657, -0.94981149,
                -0.96949489, -1.00075676, -1.02275585, -1.03896571, -1.05285987, -1.05980695, -1.06906972, -1.08412173,
                -1.08643743, -1.07485896, -1.06559618, -1.05054417, -1.05054417, -1.06559618, -1.07949035, -1.07717465,
                -1.06328049, -1.05170202, -1.04938633, -1.06212264, -1.06559618, -1.06443834, -1.06559618, -1.05517556,
                -1.05054417, -1.05054417, -1.04938633, -1.04938633, -1.05401772, -1.06675403, -1.07370111, -1.07370111,
                -1.07138542, -1.07485896, -1.09338451, -1.09106882, -1.07949035, -1.07022757, -1.06559618, -1.07138542,
                -1.08412173, -1.09222666, -1.10033159, -1.10148944, -1.09685805, -1.09685805, -1.10380513, -1.10264729,
                -1.10380513, -1.11191006, -1.1153836, -1.12811992, -1.16864456, -1.19527504, -1.19759073, -1.20106427,
                -1.20453781, -1.20453781, -1.20453781])

        multi_step_model = tf.keras.models.Sequential()
        multi_step_model.add(tf.keras.layers.LSTM(32,
                                                  return_sequences=True,
                                                  input_shape=x_train_multi.shape[-2:]))
        multi_step_model.add(tf.keras.layers.LSTM(16, activation='relu'))
        multi_step_model.add(tf.keras.layers.Dense(72))

        multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')
        multi_step_history = multi_step_model.fit(train_data_multi, epochs=EPOCHS,
                                                  steps_per_epoch=EVALUATION_INTERVAL,
                                                  validation_data=val_data_multi,
                                                  validation_steps=50)
        np.testing.assert_allclose(multi_step_history.history['loss']
                , [0.49568035870790483, 0.3477446463704109, 0.3283150506019592, 0.24120361864566803, 0.1965937300026417
                , 0.2057566329953809, 0.1978715119510889, 0.19558004647493363, 0.19830851435661315, 0.1904714184254408]
                , atol=0.001)
        np.testing.assert_allclose(multi_step_history.history['val_loss']
                , [0.30373954311013224, 0.29454170107841493, 0.24862407758831978, 0.205373055934906, 0.19932058781385423
                , 0.2122972585260868, 0.20495799109339713, 0.2037396778166294, 0.18741388574242593, 0.1875745528936386]
                , atol=0.0045)

        expected = [[0.41822052,0.39796174,0.38474515,0.38456917,0.3718312,0.37446836,0.34106544,0.37549073,0.35206997,0.34984684,0.33306637,0.35911554,0.33116972,0.35569966,0.3416351,0.3682162,0.35983977,0.36590013,0.35157388,0.36097145,0.37155426,0.36608526,0.39992002,0.4246857,0.4085364,0.45684755,0.45601943,0.4760541,0.48043686,0.5135533,0.50805616,0.5181598,0.5510976,0.55942905,0.605308,0.61467,0.6333632,0.65337,0.6841511,0.6956255,0.72729427,0.73273414,0.7474375,0.76777506,0.8056673,0.84090173,0.8425454,0.89695495,0.88671094,0.92481685,0.9465139,0.9585051,0.9738372,1.0424668,1.0419027,1.0729506,1.076236,1.1165546,1.132253,1.180511,1.1851672,1.2083395,1.2260803,1.2182425,1.2367826,1.2672977,1.2683847,1.2685986,1.3366208,1.3490661,1.3441888,1.3453659],
                    [0.86523145,0.81225824,0.83318275,0.79477894,0.76386356,0.741487,0.72911644,0.73930764,0.7173937,0.7195646,0.66925454,0.6765063,0.631583,0.65700847,0.6564428,0.6084909,0.59608227,0.5803269,0.5423741,0.5487569,0.5482944,0.52249634,0.5153425,0.5138082,0.49925265,0.5167658,0.4993508,0.4571529,0.43331033,0.48251554,0.44910395,0.39601353,0.43997124,0.39321393,0.43725345,0.42007288,0.41958702,0.41233414,0.433312,0.39986992,0.4266094,0.40804812,0.40136904,0.39682966,0.40239713,0.39788097,0.3835857,0.44325578,0.4085406,0.47633272,0.44891524,0.4333865,0.47418907,0.53959674,0.53884894,0.5326968,0.50561804,0.543646,0.5478322,0.5997139,0.54636043,0.6219896,0.57401955,0.5991538,0.62843865,0.6948869,0.6751651,0.68083864,0.7258818,0.76314557,0.75721335,0.775564],
                    [0.9938242,0.9284321,0.9822884,0.9452584,0.92036307,0.91540086,0.90169466,0.8861804,0.89985794,0.9202615,0.8920273,0.8964544,0.83469254,0.8578964,0.8717283,0.8099131,0.8303045,0.7784234,0.7671747,0.74769205,0.75809836,0.7380866,0.7392177,0.73331255,0.73314553,0.713008,0.68140274,0.6516113,0.64350605,0.620538,0.6297718,0.5528395,0.57341343,0.52088785,0.53112537,0.5227863,0.49416608,0.5020596,0.501015,0.47754744,0.45200077,0.42119956,0.42610332,0.4075387,0.3719057,0.36280817,0.33881116,0.37034327,0.32737473,0.3436821,0.3246327,0.31065205,0.28052703,0.2903543,0.30260566,0.28483236,0.2769239,0.27462178,0.2208622,0.27195477,0.22630551,0.24251741,0.21047519,0.22047946,0.26652342,0.26354924,0.24162352,0.24012612,0.21035975,0.26843342,0.24006897,0.26893008]]
        for i, e in enumerate(val_data_multi.take(3)):
            np.testing.assert_allclose(multi_step_model.predict(e[0])[0], expected[i])

        print('Done')


if __name__ == '__main__':
    unittest.main()
