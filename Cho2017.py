from scipy.io import loadmat
import numpy as np
import mne


class SubjectData:

    mat_fields = ['noise', 'rest', 'srate', 'movement_left', 'movement_right', 'movement_event', 'n_movement_trials',
                  'imagery_left', 'imagery_right', 'n_imagery_trials', 'frame', 'imagery_event', 'comment', 'subject',
                  'bad_trial_indices', 'psenloc', 'senloc']

    channel_types = ['eeg' for _ in range(64)] + ['emg' for _ in range(4)]

    # Biosemi ActiveTwo system collected with the BCI2000 system 3.0.2
    # Two EMG electrodes were attached to the flexor digitorum profundus
    # and extensor digitorum on each arm.
    # FIXME: Channels 65 - 68 EMGs need to check the mapping
    # Link to system specific electrode placement procedure
    # https://www.bci2000.org/mediawiki/index.php/User_Tutorial:EEG_Measurement_Setup
    channel_names = ['FP1', 'AF7', 'AF3',               # 1  - 3
                     'F1', 'F3', 'F5', 'F7',            # 4  - 7
                     'FT7', 'FC5', 'FC3', 'FC1',        # 8  - 11
                     'C1', 'C3', 'C5', 'T7',            # 12 - 15
                     'TP7', 'CP5', 'CP3', 'CP1',        # 16 - 19
                     'P1', 'P3', 'P5', 'P7', 'P9',      # 20 - 24
                     'PO7', 'PO3', 'O1',                # 25 - 27
                     'Iz', 'Oz', 'POz', 'Pz', 'CPZ',    # 28 - 32
                     'FPZ', 'FP2', 'AF8', 'AF4', 'AFZ', # 33 - 37
                     'FZ', 'F2', 'F4', 'F6', 'F8',      # 38 - 42
                     'FT8', 'FC6', 'FC4', 'FC2', 'FCz', # 43 - 47
                     'Cz', 'C2', 'C4', 'C6', 'T8',      # 48 - 52
                     'TP8', 'CP6', 'CP4', 'CP2',        # 53 - 56
                     'P2', 'P4', 'P6', 'P8', 'P10',     # 57 - 61
                     'PO8', 'PO4', 'O2',                # 62 - 64
                     'FDP_L', 'ED_L', 'FDP_R', 'ED_R']  # 65 - 68
    
    stim_channel = 'STI'
    
    noise_measurement_types = ['blinking', 'eye_up-down', 'eye_left-right', 'jaw', 'head_left-right']

    def __init__(self, data_file_path, verbose=False):
        self.__data = {}
        self.raw_imagery_left = None
        self.raw_imagery_right = None
        self.raw_movement_left = None
        self.raw_movement_right = None
        self.raw_noise = {}
        self.raw_rest = None
        self.__demarshal_mat(data_file_path)
        self.__generate_mne_raw(verbose=verbose)

    def __getitem__(self, item):
        """To access the original fields from the dataset .mat files prepend 'mat_' to the field name"""
        if type(item) == str and item[:4] == 'mat_':
            assert item[4:] in self.__data.keys()
            return self.__data[item[4:]]
        else:
            raise IndexError('Unknown field to access SubjectData: {}'.format(item))

    def __getattr__(self, name):
        """To access the original fields from the dataset .mat files prepend 'mat_' to the attribute name"""
        if name[:4] == 'mat_':
            assert name[4:] in self.__data.keys()
            return self.__data[name[4:]]
        else:
            raise AttributeError('Unknown attribute to access SubjectData: {}'.format(name))

    def __demarshal_mat(self, file_name):
        mat = loadmat(file_name)
        for field in mat['eeg'].dtype.fields.keys():
            assert field in self.mat_fields     # Make sure all fields are accounted for
            data = mat['eeg'][field][0][0]
            if field == 'noise':    # MatLab cell object
                num_entries = data.shape[0]
                entry_size = data[0][0].shape
                self.__data[field] = np.ndarray((num_entries, *entry_size))
                for i in range(num_entries):
                    self.__data[field][i] = data[i][0]
            elif field == 'bad_trial_indices':  # This is a MatLab struct... TODO: Not sure how to demarshal this
                self.__data[field] = data[0][0]
            elif field in ['srate', 'n_movement_trials', 'n_imagery_trials', 'comment', 'subject']:     # Scalar/strings
                while type(data) == np.ndarray:
                    data = data[0]
                str_type = str(type(data))
                if 'str' in str_type:
                    self.__data[field] = str(data)
                elif 'int' in str_type:
                    self.__data[field] = int(data)
                elif 'float' in str_type:
                    self.__data[field] = float(data)
                else:
                    raise TypeError('Unsuppored type when demarshalling field {} of value {} of type {}'.format(
                        field, data, type(data)))
            else:
                self.__data[field] = data

    def __generate_mne_raw(self, verbose=None):
        """Generate MNE.Raw from EEG measurements"""
        # Note that the original dataset gives the raw ADC valus from the Biosemi Activetwo. The conversion rate is
        # 31.25nV/bit https://www.biosemi.com/activetwo_full_specs.htm
        scale = 31.25e-9
#         self.mat_imagery_left = (self.mat_imagery_left - self.mat_imagery_left.mean(axis=1, keepdims=True)) * scale
#         self.mat_imagery_right = (self.mat_imagery_right - self.mat_imagery_right.mean(axis=1, keepdims=True)) * scale
#         self.mat_movement_left = (self.mat_movement_left - self.mat_movement_left.mean(axis=1, keepdims=True)) * scale
#         self.mat_movement_right = (self.mat_movement_right - self.mat_movement_right.mean(axis=1, keepdims=True)) * scale
        self.mat_imagery_left = self.mat_imagery_left * scale
        self.mat_imagery_right = self.mat_imagery_right * scale
        self.mat_movement_left = self.mat_movement_left * scale
        self.mat_movement_right = self.mat_movement_right * scale
        self.mat_rest = (self.mat_rest - self.mat_rest.mean(axis=1, keepdims=True)) / scale
        # Noise is 0-meaned and converted to V below
        
        # BCI2000 Electrode placement https://www.bci2000.org/mediawiki/index.php/User_Tutorial:EEG_Measurement_Setup
        # TODO: Check if montage coord frame is actually head or if we need to give fiducials
        # LPA and RPA Points might be the avg between P9/10 and T7/T8 
        senloc = self.mat_psenloc * 0.121
        # senloc = self.mat_senloc * 0.0105    # Hmmm which one?
        montage = mne.channels.make_dig_montage({k: v for k, v in zip(self.channel_names, senloc)},
                                                # lpa=senloc[self.channel_names.index('P9')],   # See TODO
                                                # rpa=senloc[self.channel_names.index('P10')],
                                                coord_frame='head')
        # montage = mne.channels.make_standard_montage('biosemi64')
        channel_names_w_stim = self.channel_names.copy()
        channel_names_w_stim.append(self.stim_channel)
        channel_types_w_stim = self.channel_types.copy()
        channel_types_w_stim.append('stim')
        info_w_stim = mne.create_info(channel_names_w_stim, self.mat_srate, channel_types_w_stim)
        self.raw_imagery_left = mne.io.RawArray(np.vstack((self.mat_imagery_left, self.mat_imagery_event)),
                                                info_w_stim, verbose=verbose)
        self.raw_imagery_right = mne.io.RawArray(np.vstack((self.mat_imagery_right, self.mat_imagery_event)),
                                                 info_w_stim, verbose=verbose)
        self.raw_imagery_left.set_montage(montage)
        self.raw_imagery_right.set_montage(montage)

        self.raw_movement_left = mne.io.RawArray(np.vstack((self.mat_movement_left, self.mat_movement_event)),
                                                 info_w_stim, verbose=verbose)
        self.raw_movement_right = mne.io.RawArray(np.vstack((self.mat_movement_right, self.mat_movement_event)),
                                                  info_w_stim, verbose=verbose)
        self.raw_movement_left.set_montage(montage)
        self.raw_movement_right.set_montage(montage)
        
        info_no_stim = mne.create_info(self.channel_names, self.mat_srate, self.channel_types)
        self.raw_rest = mne.io.RawArray(self.mat_rest, info_no_stim, verbose=verbose)
        self.raw_rest.set_montage(montage)
        for i, noise_type in enumerate(self.noise_measurement_types):
            self.mat_noise[i] = (self.mat_noise[i] - self.mat_noise[i].mean(axis=1, keepdims=True)) / scale
            self.raw_noise[noise_type] = mne.io.RawArray(self.mat_noise[i], info_no_stim, verbose=verbose)
            self.raw_noise[noise_type].set_montage(montage)
    
    def get_epochs(self, which, tmin=-2, tmax=5, filter_freqs=(8, 30), reject_criteria={'eeg': 100e-6},
                   filter_props=dict(picks=['eeg'], fir_design='firwin', skip_by_annotation='edge'),
                   **kwargs):
        assert which in ['imagery_left', 'imagery_right', 'movement_left', 'movement_right']
        # For tmin/tmax, note that each trial is 7 seconds: 2s before cue, cue @ t=0, stimulus for 3s, then 2s after          
        attr = 'raw_' + which
        raw = getattr(self, attr)
        imagery_events = mne.find_events(raw, stim_channel=self.stim_channel)
        epoch = mne.Epochs(raw, imagery_events, tmin=tmin, tmax=tmax, preload=True, **kwargs)
        
        if filter_freqs is not None:
            epoch = epoch.filter(*filter_freqs, **filter_props)
        
        if reject_criteria is not None:
            epoch = epoch.drop_bad(reject=reject_criteria)
        
        return epoch
        

if __name__ == '__main__':
    # Test usage of SubjectData class
    s01 = SubjectData('s01.mat')
    print('sampling rate', s01.srate)
    print('rest signal shape', s01['rest'].shape)

