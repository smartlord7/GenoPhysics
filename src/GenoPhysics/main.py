import pandas as pd

PATH_DATA = '../../data/'
EXTENSION_DATA = '.csv'
DELIMITER_DATA = ','
CHAR_COMMENT = '#'
PATH_NASA_EXOPLANET_DATA = PATH_DATA + \
                           'NasaExoplanetArchive_PS_2023.03.20_19.05' + \
                           EXTENSION_DATA


def load_dataset(path, delimiter=DELIMITER_DATA, comment=CHAR_COMMENT):
    dataset = pd.read_csv(path, delimiter=delimiter, comment=comment, low_memory=False)

    return dataset


def main():
    print('Loading dataset from %s...' % PATH_NASA_EXOPLANET_DATA)
    dataset = load_dataset(PATH_NASA_EXOPLANET_DATA)
    print('Done loading dataset.')

    pass


if __name__ == '__main__':
    main()
