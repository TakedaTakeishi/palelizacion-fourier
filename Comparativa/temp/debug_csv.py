import pandas as pd, glob
for f in glob.glob("Con_*.csv"):
    df = pd.read_csv(f, header=3)
    print(f, df.columns.tolist()[-5:])
