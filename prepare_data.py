import pandas as pd
from sklearn.preprocessing import LabelBinarizer, OrdinalEncoder, LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

def process(df, ordinal_features, target):
    # store target into separate dataframe before scaling
    df_target = df[target]
    # remove target feature
    df.drop(columns=target, inplace=True)
    # initialize ordinal_variables
    ordinal_variables = []
    # count number of different unique values for each feature
    df_uniques = pd.DataFrame([[i, len(df[i].unique())] for i in df.columns], columns=['Variable', 'Unique Values']).set_index('Variable')
    # retrieve binary variables/features
    binary_variables = list(df_uniques[df_uniques['Unique Values'] == 2].index)
    # retrieve categorical variables/features
    categorical_variables = list(df_uniques[(df_uniques['Unique Values'] <= 10) & (df_uniques['Unique Values'] > 2)].index)
    # retrieve ordinal variables/features
    if ordinal_features is not None:
        ordinal_variables.append(ordinal_features)
    # retrieve numeric variables/features
    numeric_variables = list(set(df.columns) - set(ordinal_variables) - set(categorical_variables) - set(binary_variables))
    # recover low features and fille NaN values
    df_numerical = clean_numerical(df[numeric_variables])
    # update main df
    df[numeric_variables].update(df_numerical)
    # encode ordinal features
    Oe = OrdinalEncoder()
    df[ordinal_variables] = Oe.fit_transform(df[ordinal_variables])
    # encode binary features
    lb = LabelBinarizer()
    for column in binary_variables:
        df[column] = lb.fit_transform(df[column])
    # encode ordinal and numeric features
    mm = MinMaxScaler()
    for column in [ordinal_variables + numeric_variables]:
        df[column] = mm.fit_transform(df[column])
    # encode categorical features
    df = pd.get_dummies(df, columns = categorical_variables, drop_first=True)
    # get categorical dummies
    categorical_dumm_variables = list(set(df.columns) - set(ordinal_variables) - set(numeric_variables) - set(binary_variables))
    # encode labels
    Le = LabelEncoder()
    df[target] = Le.fit_transform(df_target)
    return df, df[numeric_variables], df[categorical_dumm_variables], df[ordinal_variables], df[binary_variables]

def clean_numerical(df_numerical):
    # impute residual missing values
    imputer = SimpleImputer(strategy='mean')
    imputer.fit(df_numerical)
    # rebuild dataframe from numpy array
    df_numerical = pd.DataFrame(imputer.transform(df_numerical), index=df_numerical.index, columns=df_numerical.columns)

    return df_numerical

def plot_correlation_matrix(df, target):
    top_features = df.corr().abs()[target].sort_values(ascending=False).head(30)
    plt.figure(figsize=(5,10))
    sns.heatmap(top_features.to_frame(),cmap='rainbow',annot=True,annot_kws={"size": 16},vmin=0)
    plt.title("correlation matrix")
    plt.show()