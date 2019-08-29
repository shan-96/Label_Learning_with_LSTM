import pandas as pd

from model.Kfold import Kfold

kfold = Kfold(10, None)

if __name__ == "__main__":
    n = 0
    df = pd.read_csv("../resources/data.csv")
    for i in df['resolution'].unique().tolist():
        print("\'" + i + "\'" + " : " + str(n) + ",")
        n += 1
