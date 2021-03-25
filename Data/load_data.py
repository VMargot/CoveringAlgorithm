from os.path import dirname, join
import pandas as pd

target_dict = {'student_mat': 'G3',
               'student_por': 'G3',
               'student_mat_easy': 'G3',
               'student_por_easy': 'G3',
               'boston': 'MEDV',
               'bike_hour': 'cnt',
               'bike_day': 'cnt',
               'mpg': 'mpg',
               'machine': 'ERP',
               'abalone': 'Rings',
               'prostate': 'lpsa',
               'ozone': 'ozone',
               'diabetes': 'Y'}


def load_data(name: str, racine_path: str = None):
    """
    Parameters
    ----------
    name: a chosen data set
    racine_path : the racine path

    Returns
    -------
    data: a pandas DataFrame
    """
    if racine_path is None:
        racine_path = dirname(__file__)
    if 'student' in name:
        if 'student_por' in name:
            data = pd.read_csv(join(racine_path, 'Student/student-por.csv'),
                               sep=';')
        elif 'student_mat' in name:
            data = pd.read_csv(join(racine_path, 'Student/student-mat.csv'),
                               sep=';')
        else:
            raise ValueError('Not tested dataset')
        # Covering Algorithm allow only numerical features.
        # We can only convert binary qualitative features.
        data['sex'] = [1 if x == 'F' else 0 for x in data['sex'].values]
        data['Pstatus'] = [1 if x == 'A' else 0 for x in data['Pstatus'].values]
        data['famsize'] = [1 if x == 'GT3' else 0 for x in data['famsize'].values]
        data['address'] = [1 if x == 'U' else 0 for x in data['address'].values]
        data['school'] = [1 if x == 'GP' else 0 for x in data['school'].values]
        data = data.replace('yes', 1)
        data = data.replace('no', 0)

        if 'easy' not in name:
            # For an harder exercise drop G1 and G2
            data = data.drop(['G1', 'G2'], axis=1)

    elif name == 'bike_hour':
        data = pd.read_csv(join(racine_path, 'BikeSharing/hour.csv'), index_col=0)
        data = data.set_index('dteday')
    elif name == 'bike_day':
        data = pd.read_csv(join(racine_path, 'BikeSharing/day.csv'), index_col=0)
        data = data.set_index('dteday')
    elif name == 'mpg':
        data = pd.read_csv(join(racine_path, 'MPG/mpg.csv'))
    elif name == 'machine':
        data = pd.read_csv(join(racine_path, 'Machine/machine.csv'))
    elif name == 'abalone':
        data = pd.read_csv(join(racine_path, 'Abalone/abalone.csv'))
    elif name == 'ozone':
        data = pd.read_csv(join(racine_path, 'Ozone/ozone.csv'))
    elif name == 'prostate':
        data = pd.read_csv(join(racine_path, 'Prostate/prostate.csv'), index_col=0)
    elif name == 'diabetes':
        data = pd.read_csv(join(racine_path, 'Diabetes/diabetes.csv'), index_col=0)
    elif name == 'boston':
        from sklearn.datasets import load_boston
        boston_dataset = load_boston()
        data = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
        data['MEDV'] = boston_dataset.target
    else:
        raise ValueError('Not tested dataset')

    return data.dropna()
