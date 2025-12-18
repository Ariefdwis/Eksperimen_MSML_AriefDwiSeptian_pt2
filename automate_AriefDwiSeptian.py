import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import os

def main():
    print("Memulai proses otomatisasi data preprocessing...")

    
    file_path = 'dataset_raw/water_potability.csv'

    df = pd.read_csv(file_path)
    print(f"Data berhasil dimuat. Ukuran awal: {df.shape}")



    print("Melakukan imputasi data kosong...")
    imputer = SimpleImputer(strategy='mean')
    df_clean_values = imputer.fit_transform(df)
    df_clean = pd.DataFrame(df_clean_values, columns=df.columns)


    print("Melakukan scaling fitur...")
    X = df_clean.drop('Potability', axis=1)
    y = df_clean['Potability']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)


    df_final = pd.concat([X_scaled_df, y], axis=1)


    output_folder = 'dataset_preprocessing'
    output_file = 'water_potability_clean.csv'
    
    os.makedirs(output_folder, exist_ok=True)
    df_final.to_csv(f'{output_folder}/{output_file}', index=False)

    print(f"Sukses! Data bersih tersimpan di: {output_folder}/{output_file}")

if __name__ == "__main__":
    main()