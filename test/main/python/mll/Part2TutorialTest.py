import unittest

import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from mll.LSTMPredictor import LSTMPredictor

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
                                   , [0.5900344, 0.64682704, 1.3358905])

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
                , [0.30896712981164454, 0.26240251809358595, 0.2616553147137165, 0.25689387537539005, 0.2264231711626053, 0.24175462897755848, 0.24145282685756683, 0.2408904529362917, 0.24501676440238954, 0.23879031479358673]
                , atol=0.001)
        np.testing.assert_allclose(single_step_history.history['val_loss']
                , [0.2646432928740978, 0.2445167075097561, 0.24731518439948558, 0.24523079961538316, 0.23587791658937932, 0.26497877776622775, 0.2581120006740093, 0.23956751435995102, 0.24788874626159668, 0.25179353058338166]
                , atol=0.0045)
        np.testing.assert_allclose([single_step_model.predict(x)[0][0] for x, _ in val_data_single.take(3)]
                                   , [1.4486389, 0.7135288, 0.1295506])
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
            np.testing.assert_allclose(y[0].numpy(), [-0.9648635, -0.90118192, -0.88497207, -0.86065728, -0.85834159, -0.86181513, -0.8537102, -0.82939542, -0.85023666, -0.90118192, -0.93707518, -0.94170657, -0.94981149, -0.96949489, -1.00075676, -1.02275585, -1.03896571, -1.05285987, -1.05980695, -1.06906972, -1.08412173, -1.08643743, -1.07485896, -1.06559618, -1.05054417, -1.05054417, -1.06559618, -1.07949035, -1.07717465, -1.06328049, -1.05170202, -1.04938633, -1.06212264, -1.06559618, -1.06443834, -1.06559618, -1.05517556,-1.05054417, -1.05054417, -1.04938633, -1.04938633, -1.05401772, -1.06675403, -1.07370111, -1.07370111, -1.07138542, -1.07485896, -1.09338451, -1.09106882, -1.07949035, -1.07022757, -1.06559618, -1.07138542, -1.08412173, -1.09222666, -1.10033159, -1.10148944, -1.09685805, -1.09685805, -1.10380513, -1.10264729, -1.10380513, -1.11191006, -1.1153836, -1.12811992, -1.16864456, -1.19527504, -1.19759073, -1.20106427, -1.20453781, -1.20453781, -1.20453781])

        multi_step_model = tf.keras.models.Sequential()
        multi_step_model.add(tf.keras.layers.LSTM(32,
                                                  return_sequences=True,
                                                  input_shape=x_train_multi.shape[-2:]))
        multi_step_model.add(tf.keras.layers.LSTM(16)) #without activation='relu' to use CuDNNLSTM https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM
        # multi_step_model.add(tf.keras.layers.LSTM(16, activation='relu'))
        multi_step_model.add(tf.keras.layers.Dense(72))

        multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')
        multi_step_history = multi_step_model.fit(train_data_multi, epochs=EPOCHS,
                                                  steps_per_epoch=EVALUATION_INTERVAL,
                                                  validation_data=val_data_multi,
                                                  validation_steps=50)
        np.testing.assert_allclose(multi_step_history.history['loss']
                , [0.4450967706739902, 0.2809910763800144, 0.2509454064071178, 0.22688977420330048, 0.19398607790470124, 0.20107007094511245, 0.19687376536428927, 0.1952586491405964, 0.1979307834804058, 0.18740502052009106]
                , atol=0.001)
        np.testing.assert_allclose(multi_step_history.history['val_loss']
                , [0.28981183111667635, 0.23509616151452065, 0.22503124944865704, 0.199942394644022, 0.19537991806864738, 0.2210526643693447, 0.20746466845273973, 0.22165924802422524, 0.19337186850607396, 0.18313108265399933]
                , atol=0.0045)

        expected = [[0.42008147,0.4074365,0.4190545,0.38092047,0.37764043,0.3532061,0.3635833,0.353794,0.3636174,0.36912882,0.34708217,0.3492923,0.3271138,0.34136534,0.3694318,0.337804,0.36689726,0.35643876,0.35860327,0.36421573,0.3855072,0.40817818,0.39922044,0.41137493,0.4444736,0.443277,0.46242836,0.47056186,0.47642902,0.49946585,0.5162472,0.5412186,0.5675696,0.5948358,0.60343724,0.611224,0.66957885,0.65411466,0.6697624,0.6971751,0.74489814,0.7621752,0.7942886,0.8095534,0.79381627,0.84694064,0.87268865,0.9075441,0.90722024,0.915903,0.95743406,0.9836918,0.9889545,1.0059024,1.0255847,1.0458528,1.0805799,1.0764143,1.1024889,1.1363546,1.1611388,1.1497985,1.1532031,1.181525,1.2072618,1.232615,1.1955132,1.2199786,1.2436962,1.2822646,1.2872165,1.2864743],
                    [0.7812743,0.8194141,0.7613482,0.75587755,0.7333431,0.7407532,0.7246601,0.68898964,0.67698884,0.624457,0.6344327,0.6027515,0.6476809,0.58828735,0.5657572,0.57673526,0.57208455,0.5547002,0.5476473,0.5289162,0.53235847,0.5214933,0.4908832,0.49427837,0.45242396,0.46084976,0.46418008,0.42138994,0.43918836,0.42978412,0.41351685,0.43037,0.39922905,0.4408551,0.3923477,0.4042978,0.40676972,0.38691247,0.39387703,0.37547672,0.3900192,0.46217522,0.42298168,0.441446,0.41992244,0.43954876,0.46803075,0.43839237,0.44874194,0.46782932,0.45921138,0.46045873,0.49706095,0.52588063,0.49481007,0.5131276,0.54101694,0.5282164,0.5888023,0.59030914,0.5973701,0.60163265,0.6296886,0.6582335,0.65569896,0.6550842,0.7084278,0.70974785,0.7236915,0.73276305,0.76109904,0.74581313],
                    [0.97796416,0.94317424,0.973258,0.93881714,0.9341861,0.9135064,0.92150766,0.8993654,0.9100298,0.8901999,0.8815167,0.87288153,0.84119916,0.84060353,0.85510087,0.82818663,0.7997056,0.79429907,0.76427853,0.74651986,0.74829835,0.7466254,0.71017355,0.7084104,0.7050966,0.68472457,0.655819,0.6620807,0.6470221,0.61601824,0.5927886,0.58686787,0.58980066,0.552134,0.5391422,0.53496736,0.5190323,0.5000161,0.4889864,0.4723783,0.4615299,0.4421502,0.42290407,0.39695793,0.3901872,0.389645,0.35229707,0.36673546,0.35595116,0.3203716,0.3019779,0.3134658,0.2953146,0.26904568,0.2680464,0.24839607,0.2376272,0.2402171,0.22638237,0.20988297,0.21445316,0.22646923,0.20198339,0.20088762,0.18600768,0.18041697,0.15956028,0.18527961,0.19319332,0.1862053,0.16438788,0.1657778]]
        for i, e in enumerate(val_data_multi.take(3)):
            np.testing.assert_allclose(multi_step_model.predict(e[0])[0], expected[i])

        print('Done')

    def test_multi_step_model(self):
        TRAIN_SPLIT = 300000
        BATCH_SIZE = 256
        BUFFER_SIZE = 10000
        EVALUATION_INTERVAL = 200
        EPOCHS = 10
        past_history = 720
        STEP = 6
        future_target = 72
        tf.random.set_seed(13)
        zip_path = tf.keras.utils.get_file(
            origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
            fname='jena_climate_2009_2016.csv.zip',
            extract=True)
        csv_path, _ = os.path.splitext(zip_path)
        df = pd.read_csv(csv_path).set_index('Date Time')

        predictor = LSTMPredictor(df[['p (mbar)', 'T (degC)', 'rho (g/m**3)']], 'T (degC)', TRAIN_SPLIT, past_history
                                  , future_target, STEP, BATCH_SIZE, BUFFER_SIZE, EVALUATION_INTERVAL, EPOCHS)

        val_data_multi = predictor.validation_dataset
        train_data_multi = predictor.train_data_multi

        for x, y in train_data_multi.take(1):
            np.testing.assert_allclose(y[0].numpy(),[-0.3928871311805136, -0.40678129406519214, -0.42183330385692724, -0.4357274667416058, -0.4519373234403974, -0.46004225178979324, -0.48204134302386753, -0.5028825873508854, -0.5225659847708467, -0.5237238316779033, -0.5202502909567336, -0.52140813786379, -0.529513066213186, -0.5248816785849597, -0.5248816785849597, -0.5306709131202425, -0.5318287600272991, -0.5341444538414121, -0.5630906265178257, -0.6105623497071442, -0.6175094311494833, -0.6244565125918227, -0.6348771347553316, -0.644139910011784, -0.6510869914541233, -0.6534026852682363, -0.6406663692906143, -0.605930962078918, -0.5908789522871829, -0.6047731151718614, -0.6325614409412186, -0.6151937373353703, -0.595510339915409, -0.5966681868224656, -0.6128780435212573, -0.6001417275436353, -0.6128780435212573, -0.6221408187777097, -0.6221408187777097, -0.6198251249635965, -0.5769847894025043, -0.5491964636331472, -0.5654063203319389, -0.5654063203319389, -0.559617085796656, -0.5341444538414121, -0.5480386167260907, -0.567722014146052, -0.5746690955883913, -0.5793004832166173, -0.5723534017742781, -0.5619327796107693, -0.5457229229119777, -0.5457229229119777, -0.5688798610531085, -0.5758269424954477, -0.580458330123674, -0.5816161770307305, -0.5793004832166173, -0.5862475646589567, -0.5862475646589567, -0.5908789522871829, -0.6105623497071442, -0.6406663692906143, -0.6557183790823493, -0.6638233074317452, -0.6649811543388018, -0.6811910110375934, -0.7020322553646112, -0.7205578058775159, -0.7332941218551379, -0.7564510599962688])

        multi_step_model = predictor.multi_step_model
        multi_step_history = predictor.multi_step_history

        np.testing.assert_allclose(multi_step_history.history['loss']
                , [0.44376814126968384, 0.2823571302741766, 0.2520401889830828, 0.22650205560028552, 0.19273263037204744, 0.2003172218144595, 0.19582362338900566, 0.19108724601566793, 0.19595367684960366, 0.1866853316873312]
                , atol=0.001)
        np.testing.assert_allclose(multi_step_history.history['val_loss']
                , [0.2699364010989666, 0.2250477008521557, 0.2349950946867466, 0.2081570640951395, 0.1943892778456211, 0.21836476922035217, 0.20557487800717353, 0.19481154128909112, 0.18428097143769265, 0.18299187153577803]
                , atol=0.0045)

        expected = [[0.46694633, 0.48803085, 0.44418874, 0.46234182, 0.48076704, 0.42408413, 0.43972373, 0.48175192, 0.44634333, 0.4145841, 0.4160986, 0.39468956, 0.44700748, 0.4049536, 0.41337106, 0.4255219, 0.39744985, 0.41615492, 0.44762936, 0.43669233, 0.4423308, 0.38252208, 0.42939943, 0.43539235, 0.44823453, 0.4364457, 0.43964478, 0.5068632, 0.43076795, 0.44051552, 0.4631386, 0.48653477, 0.5413683, 0.5316197, 0.5568972, 0.5305403, 0.54020596, 0.5961387, 0.60955983, 0.6264573, 0.6351678, 0.6678768, 0.6749518, 0.7119457, 0.726905, 0.74692106, 0.7213109, 0.7571528, 0.76886123, 0.78013337, 0.78139734, 0.83400244, 0.8240561, 0.8714807, 0.9007517, 0.88693297, 0.8627477, 0.9171753, 0.9519368, 0.93352854, 0.9481875, 0.9717384, 1.0033365, 0.94051194, 0.98623735, 1.0357206, 1.0420145, 0.9752281, 1.0669636, 1.0238562, 1.0707637, 1.0852065],
                    [0.85909295, 0.8413977, 0.82436556, 0.8307964, 0.8362157, 0.77573717, 0.7789502, 0.79811645, 0.77215743, 0.74277425, 0.70310235, 0.6983324, 0.71356493, 0.6763814, 0.6626318, 0.64570004, 0.61624694, 0.6269574, 0.6190287, 0.6066154, 0.59887046, 0.53993875, 0.5495325, 0.5405249, 0.52947176, 0.53324777, 0.5067998, 0.5235339, 0.46107313, 0.4248848, 0.4431189, 0.47040075, 0.4779082, 0.44475812, 0.4401222, 0.4346435, 0.4154048, 0.41574115, 0.4406949, 0.4412855, 0.41598785, 0.42245552, 0.43287164, 0.44032657, 0.421, 0.4372495, 0.43661344, 0.43308744, 0.4463272, 0.43778417, 0.41866225, 0.44715646, 0.44035247, 0.47177345, 0.49821392, 0.4630729, 0.48153225, 0.49711525, 0.52242595, 0.5161866, 0.525907, 0.5256296, 0.5621478, 0.5206258, 0.5708633, 0.6137809, 0.6262361, 0.57610756, 0.6545731, 0.6235799, 0.65283865, 0.6714184],
                    [0.90458065, 0.90268266, 0.9080104, 0.91603845, 0.8950523, 0.8645335, 0.87558293, 0.8565853, 0.87149405, 0.83764315, 0.8245188, 0.8084488, 0.8145884, 0.800333, 0.7867489, 0.76631147, 0.7336164, 0.7350821, 0.7340874, 0.7127439, 0.7059335, 0.69394547, 0.67742765, 0.64781857, 0.63943577, 0.62577534, 0.62245435, 0.61346716, 0.59348726, 0.5450138, 0.5493898, 0.55728334, 0.53911453, 0.50500834, 0.50296324, 0.48283136, 0.44784462, 0.42954504, 0.4430745, 0.4424591, 0.39669335, 0.39480194, 0.3883809, 0.37953082, 0.37222755, 0.34149852, 0.32841474, 0.31415883, 0.30783215, 0.31005904, 0.30709675, 0.28710195, 0.27267593, 0.2731855, 0.23967947, 0.23528147, 0.23580705, 0.22802968, 0.2216954, 0.21129617, 0.20448238, 0.20718783, 0.20824051, 0.18269664, 0.19593517, 0.18262216, 0.17959888, 0.19406661, 0.1961084, 0.19404104, 0.1973263, 0.18274698]]

        for i, e in enumerate(val_data_multi.take(3)):
            np.testing.assert_allclose(multi_step_model.predict(e[0])[0], expected[i])


if __name__ == '__main__':
    unittest.main()
