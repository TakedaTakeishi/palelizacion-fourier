import pandas as pd
excel_path = "6BV1_Luna_Gonzales_Fase2.xlsx"
df = pd.read_excel(excel_path, sheet_name="Fourier", header=3)
print("Columnas:")
print(df.columns.tolist())
print("\nPrimeras filas:")
print(df.head())
