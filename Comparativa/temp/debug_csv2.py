import pandas as pd, glob
for f in glob.glob("Con_*.csv"):
    # Probar varias formas
    df = pd.read_csv(f, skiprows=3, header=0)
    print(f"skiprows=3: {df.shape} | cols: {df.columns.tolist()[-5:]}")
    df2 = pd.read_csv(f, skiprows=4, header=None)
    print(f"skiprows=4: {df2.shape} | last cols sample: {df2.iloc[0, -5:].tolist()}")
    print()
