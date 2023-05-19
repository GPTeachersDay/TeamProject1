from modules.Abalone import Modeling

ref_model = Modeling()
df, X, y, X_train, X_test, y_train, y_test = ref_model.load_data()
print(df)
