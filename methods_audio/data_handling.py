import tensorflow as tf
import tensorflow_io as tfio
import os 


def get_data():
    # Data from the elephant listening project 
    general_path = os.path.join('data', 'Clips')

    # To ensure that both classes have same of samples and to increase the number of gunshots, 
    # I extracted extra data from: https://data.mendeley.com/datasets/x48cwz364j/3 
    background_path = os.path.join('data', 'Sounds_background')
    guns_path = os.path.join('data', 'Sounds_gunshots')

    gunshot_files = [os.path.join(general_path, 'pnnn*'), os.path.join(general_path, 'ecoguns*'), os.path.join(guns_path, '*\.wav')]

    no_gunshot_files = [os.path.join(general_path, 'other*'), os.path.join(background_path, '*\.wav')] 
    gunshot = tf.data.Dataset.list_files(gunshot_files, shuffle=False)  # setting shuffle to False so that the files get always read in the same order 
    no_gunshot = tf.data.Dataset.list_files(no_gunshot_files, shuffle=False) # setting shuffle to False so that the files get always read in the same order 

    #to see how many files are in each group: 
    #num_elements = tf.data.experimental.cardinality(no_gunshot).numpy()

    # Add labels to the data 
    gunshot = tf.data.Dataset.zip((gunshot, tf.data.Dataset.from_tensor_slices(tf.ones(len(gunshot)))))
    no_gunshot= tf.data.Dataset.zip((no_gunshot, tf.data.Dataset.from_tensor_slices(tf.zeros(len(no_gunshot)))))

    # Concatenate gunshots and no gunshots and shuffle data 
    data = gunshot.concatenate(no_gunshot)
    data = data.cache()
    data = data.shuffle(buffer_size=1000, seed = 123) # mixing training samples 1000 at the time  

    return data


def read_in_data(file_name, label): 
    file_contents = tf.io.read_file(file_name) #retuns a string 
    wave, _ = tf.audio.decode_wav(file_contents, desired_channels=1) # transforms string into actual wav
    wave = wave - tf.reduce_mean(wave) # remove the mean 
    wave = wave / tf.reduce_max(tf.abs(wave)) #normalize 
    wave = tf.squeeze(wave, axis= -1) #removes axis 
    return wave, label

def extract_samples_labels(data): 
    iterator = data.as_numpy_iterator()
    x = []
    y = []
    while True:
        try: 
            x_temp, y_temp = iterator.next()
            x.append(x_temp)
            y.append(y_temp)
        except Exception:
            break 
    return x,y 


def convert_to_spectogram(wave):
     # Create spectogram 
    # 1. Fast fourier transform 
    spectrogram = tf.signal.stft(wave, frame_length=256, frame_step=128)  # Paper: 'Automated detection of gunshots in tropical forests using CNN' 
    # frame_length =  window length in samples
    # frame_step = number of samples to step
    # 'Time frequency compromise' 
    # if window size is small: you get good time resolution in exchange of poor frequency resolution 

    # 2. Obtain the magnitude of the STFT
    spectrogram = tf.abs(spectrogram)

    # 3. Tranform it into appropiate format for deep learning model by adding the channel dimension (in this case 1)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram

def convert_to_mel_spectogram(wave):
    # 1. Fast fourier transform 
    spectrogram = tf.signal.stft(wave, frame_length=256, frame_step=128)  # Paper: 'Automated detection of gunshots in tropical forests using CNN' 
    # 2. Obtain the magnitude of the STFT
    spectrogram = tf.abs(spectrogram)
    # 3. Convert to mel-spectogram 
    mel_spectrogram = tfio.audio.melscale(spectrogram, rate=16000, mels=128, fmin=0, fmax=4000) # TODO: play with this numbers 

    # 4. Tranform it into appropiate format for deep learning model by adding the channel dimension
    mel_spectrogram = tf.expand_dims(mel_spectrogram, axis=2)
    return mel_spectrogram

def convert_to_mel_spectogram_db_scale(wave):
    # 1. Fast fourier transform 
    spectrogram = tf.signal.stft(wave, frame_length=256, frame_step=128)  # Paper: 'Automated detection of gunshots in tropical forests using CNN' 
    # 2. Obtain the magnitude of the STFT
    spectrogram = tf.abs(spectrogram)

    # 3. Convert into mel-spectogram
    mel_spectrogram = tfio.audio.melscale(spectrogram, rate=16000, mels=128, fmin=0, fmax=4000) #TODO: play with this numbers 

    # 4. Convert the mel-spectogram into db scale
    dbscale_mel_spectrogram = tfio.audio.dbscale(mel_spectrogram, top_db=80) 

    # 5. Tranform it into appropiate format for deep learning model by adding the channel dimension
    dbscale_mel_spectrogram = tf.expand_dims(dbscale_mel_spectrogram, axis=2)
    return dbscale_mel_spectrogram


# Function pads the clips such that they all have the same length and it converts them into the specified 'type'. 
def preprocess(file_path, label, type): 
    """This function pads all clips so that their length is 10 seconds and it transforms them to eithter: spectogram, mel spectogram or 
    mel spectogram in db scale. 

    Args:
        file_path (string)
        label (int): 1 or 0, depending if it's a gunshot or not. 
        type (string): spectogram, mel_spectogram, mel_spectogram_db

    Returns:
        int: The sum of the two numbers. #TODO 
    """

    # Load data
    wave, sr = load_data(file_path)
    max_lenght = 80000 # = 10* 8000, this means 10 seconds 

    # Padding 
    wave = wave[:max_lenght] #grab first elements up to max(lengths)
    zero_padding = tf.zeros(max_lenght - tf.shape(wave), dtype=tf.float32) # pad with zeros what doesn't meet full length 
    wave = tf.concat([zero_padding, wave],0) 

    # Transform data into specified 'type'
    if type == 'spectogram':
        new_data = convert_to_spectogram(wave)
    elif type == 'mel_spectogram': 
        new_data = convert_to_mel_spectogram(wave)
    elif type == 'mel_spectogram_db': 
        new_data = convert_to_mel_spectogram_db_scale(wave)

    return new_data, label

