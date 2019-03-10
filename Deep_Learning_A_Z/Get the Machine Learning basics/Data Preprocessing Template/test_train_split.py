# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as pyplot
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("Data.csv")
# Matrix of features
x = dataset.iloc[:, :-1].values
# IV array
y = dataset.iloc[:, -1].values

# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = "mean")
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
# Encoding the Independent Variable
labelencoder_x = LabelEncoder()
x[:, 0] = labelencoder_x.fit_transform(x[:, 0])

# New way to encode features
transformerX = ColumnTransformer(
    [(
        "dummy_colX",
        OneHotEncoder(categories = "auto"),
        [0]
    )], remainder = "passthrough"
)
x = transformerX .fit_transform(x)
x = x.astype(float)

# Future Deprecated way to encode features
# onehotencoder = OneHotEncoder(categorical_features = [0])
# x = onehotencoder.fit_transform(x).toarray()

# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

print(x)
print(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)