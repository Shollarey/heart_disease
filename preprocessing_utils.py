import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocessing(data, scaler=None, fit_scaler=True):
    df = data.copy()
    
    # Derived features
    df['age_yr'] = df['age'] / 365.25
    df['bmi'] = df['weight'] / (df['height']/100)**2
    df['map'] = (2/3 * df['ap_lo']) + (1/3 * df['ap_hi'])
    df['pp'] = df['ap_hi'] - df['ap_lo']
    df['lifestyle'] = df['active'] - (df['smoke'] + df['alco'])

    # Metabolic syndrome (X syndrome)
    df['x_syndrome'] = (
        (df['cholesterol'] > 1) &
        (df['gluc'] > 1) &
        ((df['ap_hi'] > 130) | (df['ap_lo'] > 85))
    ).astype(int)

    num_features = ['age_yr', 'height', 'weight', 'ap_hi', 'ap_lo', 
                    'bmi', 'pp', 'lifestyle', 'map']

    if fit_scaler:
        scaler = StandardScaler()
        df[num_features] = scaler.fit_transform(df[num_features])
    else:
        df[num_features] = scaler.transform(df[num_features])

    # Drop unnecessary columns
    df = df.drop(columns=['age'], errors='ignore')

    return df, scaler
