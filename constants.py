DATA_URL = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
DATA_COLUMNS = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']
NORMALIZE = False
TARGET_VARIABLE = 'MPG'
# FEATURES_TO_USE = [ 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Europe', 'Japan', 'USA']
FEATURES_TO_USE = [ 'MPG', 'Horsepower']
NORMALIZE_HORSEPOWER = False
RESULTS_PATH = "Results/"