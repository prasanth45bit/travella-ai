import pandas as pd
from model import train_model

if __name__ == "__main__":
    df = pd.read_csv("./data/travel_data.csv")
    model, pre = train_model(df, epochs=25, lr=0.001)

