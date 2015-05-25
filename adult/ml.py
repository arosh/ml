# coding: utf-8
import pandas as pd
d = pd.read_csv(
    'adult.data', sep=', ',
    names=('age', 'workclass', 'fnlwgt', 'education', 'education_num',
           'marital_status', 'occupation', 'relationship', 'race', 'sex',
           'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'label'),
    engine='python')
