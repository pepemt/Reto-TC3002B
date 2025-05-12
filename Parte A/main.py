from preprocessing_utils import preprocess_text_df
import pandas as pd

path = './Parte A/data_train.xlsx'
df = pd.read_excel(path)
df = preprocess_text_df(df)
df.to_excel("./Parte A/data_train_cleaned.xlsx", index=False)
print("Data cleaned and saved to data_train_cleaned.xlsx")