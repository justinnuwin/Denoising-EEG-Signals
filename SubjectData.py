import scipy.io
import numpy as np


class SubjectData:

    fields = ['noise', 'rest', 'srate', 'movement_left', 'movement_right', 'movement_event', 'n_movement_trials',
              'imagery_left', 'imagery_right', 'n_imagery_trials', 'frame', 'imagery_event', 'comment', 'subject',
              'bad_trial_indices', 'psenloc', 'senloc']

    def __init__(self, data_file_path):
        self.__data = {}
        self.demarshal_mat(data_file_path)

    def __getitem__(self, item):
        assert item in self.__data.keys()
        return self.__data[item]

    def __getattr__(self, name):
        assert name in self.__data.keys()
        return self.__data[name]

    def demarshal_mat(self, file_name):
        mat = scipy.io.loadmat(file_name)
        for field in mat['eeg'].dtype.fields.keys():
            assert field in self.fields     # Make sure all fields are accounted for
            data = mat['eeg'][field][0][0]
            if field == 'noise':    # MatLab cell object
                num_entries = data.shape[0]
                entry_size = data[0][0].shape
                self.__data[field] = np.ndarray((num_entries, *entry_size))
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


if __name__ == '__main__':
    # Test usage of SubjectData class
    s01 = SubjectData('s01.mat')
    print('sampling rate', s01.srate)
    print('rest signal shape', s01['rest'].shape)
    
    # Plot electrode locations
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(s01.psenloc[:, 0], s01.psenloc[:, 1], s01.psenloc[:, 2])
    fig.suptitle('psenloc')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(s01.senloc[:, 0], s01.senloc[:, 1], s01.senloc[:, 2])
    fig.suptitle('senloc')
    plt.show()
    print('Note the difference in scale between the plots')

