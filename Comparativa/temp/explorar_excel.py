import pandas as pd
import openpyxl

# Explorar el archivo Excel
excel_path = "6BV1_Luna_Gonzales_Fase2.xlsx"
xl = pd.ExcelFile(excel_path)
print("Hojas encontradas:", xl.sheet_names)

for sheet in xl.sheet_names:
    df = pd.read_excel(excel_path, sheet_name=sheet, header=None)
    print(f"\n--- Hoja: {sheet} ---")
    print(f"Dimensiones: {df.shape}")
    print(df.head(10))
