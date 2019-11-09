"""
This script extracts features from the input musical samples and exports 
the features to CSV.
"""
#%%
import os
import librosa
import pandas as pd


#%% Defining utility functions and initializing the ouput data dictionary
def get_avg_variance(coeffs_time_series):
    """
    Computes the average and variance of the input feature's
    time series. If the feature is multidimensional (ex: MFCC),
    the result is averaged over all the dimensions.
    """
    global_avg = 0
    global_var = 0
    n_dimensions, times = coeffs_time_series.shape
    for dim in range(n_dimensions):
        avg_dim = 0
        var_dim = 0
        for time in range(times):
            avg_dim += coeffs_time_series[dim][time] / times
            var_dim += ((coeffs_time_series[dim][time] - avg_dim)**2) / times
        global_avg += avg_dim / n_dimensions
        global_var += var_dim / n_dimensions
    return global_avg, global_var


OUTPUT_DATA = {}


def append_feature_to_data(output_data_dict_key, feature):
    """
    Appends the given feature (avg and variance) to the output data dict's
    given key.
    """
    OUTPUT_DATA[output_data_dict_key].append(feature[0])
    OUTPUT_DATA[output_data_dict_key].append(feature[1])


GENRES = [
    'blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop',
    'reggae', 'rock'
]

PATH_TO_DATA = os.path.dirname(__file__) + "/../genres/"

#%% Loading the 100 audio samples for each of the 10 genres and computing
# the relevant features
for i in range(10):
    for j in range(100):
        print(GENRES[i] + " file #" + str(j))
        if j < 10:
            audio, sr = librosa.load(PATH_TO_DATA + '/' + GENRES[i] + '/' +
                                     GENRES[i] + '.' + '0000' + str(j) + '.wav')
        else:
            audio, sr = librosa.load(PATH_TO_DATA + '/' + GENRES[i] + '/' +
                                     GENRES[i] + '.' + '000' + str(j) + '.wav')
        len_sample = len(audio)
        """
        Splitting the audio sample in 8 equal subsamples :
        the features will be extracted on each subsample and each will be
        classified. The final class to assign to the original sample will
        be the result of the majority vote between the 8 subsamples.
        """
        subsamples = [
            audio[int(len_sample * 0.1 * i):(int(len_sample * 0.1 * i) +
                                             int(3.5 * sr))]
            for i in range(1, 9)
        ]
        window_size = int(0.02 * sr)

        for k, subsample in enumerate(subsamples):
            # subsample 3 of the 30th blues sample will be at key "blues29_2"
            key = GENRES[i] + str(j) + "_" + str(k)
            OUTPUT_DATA[key] = [GENRES[i]]

            tempo, _ = librosa.beat.beat_track(y=subsample, sr=sr)
            OUTPUT_DATA[key].append(tempo)

            MFCCs = librosa.feature.mfcc(y=subsample,
                                         sr=sr,
                                         n_fft=window_size,
                                         n_mfcc=20)
            append_feature_to_data(key, get_avg_variance(MFCCs))

            SC = librosa.feature.spectral_centroid(y=subsample,
                                                   sr=sr,
                                                   n_fft=window_size,
                                                   freq=None)
            append_feature_to_data(key, get_avg_variance(SC))

            SB = librosa.feature.spectral_bandwidth(y=subsample,
                                                    sr=sr,
                                                    n_fft=window_size,
                                                    freq=None,
                                                    norm=True,
                                                    p=2)
            append_feature_to_data(key, get_avg_variance(SB))

            SRO = librosa.feature.spectral_rolloff(y=subsample,
                                                   sr=sr,
                                                   n_fft=window_size,
                                                   freq=None,
                                                   roll_percent=0.85)
            append_feature_to_data(key, get_avg_variance(SRO))

            SCO = librosa.feature.spectral_contrast(y=subsample,
                                                    sr=sr,
                                                    n_fft=window_size,
                                                    freq=None,
                                                    fmin=200.0,
                                                    n_bands=6,
                                                    quantile=0.02,
                                                    linear=False)
            append_feature_to_data(key, get_avg_variance(SCO))

            SF = librosa.feature.spectral_flatness(y=subsample,
                                                   n_fft=window_size,
                                                   amin=1e-10,
                                                   power=2.0)
            append_feature_to_data(key, get_avg_variance(SF))

            ZCR = librosa.feature.zero_crossing_rate(y=subsample,
                                                     frame_length=window_size,
                                                     center=True)
            append_feature_to_data(key, get_avg_variance(ZCR))

#%% Converting the features dict to a normalized Pandas dataframe and exporting
# to CSV.
FEATURES_LIST = [
    'Genre', 'Tempo', 'MFCC average', 'MFCC variance',
    'Spectral centroid average', 'Spectral centroid variance',
    'Spectral bandwidth average', 'Spectral bandwidth variance',
    'Spectral roll-off average', 'Spectral roll-off variance',
    'Spectral contrast average', 'Spectral contrast variance',
    'Spectral flatness average', 'Spectral flatness variance',
    'Zero crossing rate average', 'Zero crossing rate variance'
]

DF = pd.DataFrame.from_dict(OUTPUT_DATA, orient='index', columns=FEATURES_LIST)

DF_WITHOUT_GENRE = DF.drop(columns='Genre')
# Normalization
SCALED_DF = (DF_WITHOUT_GENRE -
             DF_WITHOUT_GENRE.mean()) / DF_WITHOUT_GENRE.std()
SCALED_DF['Genre'] = DF['Genre']

print(SCALED_DF.head())

SCALED_DF.to_csv(os.path.dirname(__file__) + '/extracted_features.csv')
