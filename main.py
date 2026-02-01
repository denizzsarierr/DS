import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error



data = pd.read_csv("train.csv")

print(data.shape)
print(data.head())
print(data.info())


edit_needed_columns = ["Alley","MasVnrArea","Electrical","FireplaceQu","GarageType","GarageFinish","GarageQual","GarageCond",
                       "GarageYrBlt","LotFrontage","MasVnrType","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","PoolQC",
                       "Fence","MiscFeature"]


edit_needed_numerical = [col for col in edit_needed_columns if data[col].dtype in ["float64","int64"]]
edit_needed_cat= [col for col in edit_needed_columns if data[col].dtype == "object"]

ordinal_quality_columns = ["ExterQual","ExterCond","BsmtQual","BsmtCond","HeatingQC","KitchenQual","FireplaceQu","GarageQual","GarageCond","PoolQC"]

nominal_cols = ["MSZoning","Street","Alley","LotShape","LandContour","Utilities",
                "LotConfig","Neighborhood","Condition1","Condition2","BldgType",
                "HouseStyle","RoofStyle","RoofMatl","Exterior1st","Exterior2nd",
                "MasVnrType","Foundation","Heating","CentralAir","Electrical",
                "GarageType","PavedDrive","Fence","MiscFeature","SaleType",
                "SaleCondition","LandSlope","BsmtExposure","BsmtFinType1",""
                "BsmtFinType2","Functional","GarageFinish"]

def clean_data(data):


    # Dropping identifier, does not help

    data.drop("Id",axis=1,inplace =True)

    # Converting NaN to None, they are meaningful, means fature does not exist.
    garage_cols = ["GarageType","GarageFinish","GarageQual","GarageCond"]
    for col in garage_cols:
        data[col] = data[col].fillna("None")
    data["MasVnrType"] = data["MasVnrType"].fillna("None")
    data["BsmtQual"] = data["BsmtQual"].fillna("None")
    data["BsmtCond"] = data["BsmtCond"].fillna("None")
    data["BsmtExposure"] = data["BsmtExposure"].fillna("None")
    data["BsmtFinType1"] = data["BsmtFinType1"].fillna("None")
    data["BsmtFinType2"] = data["BsmtFinType2"].fillna("None")
    data["PoolQC"] = data["PoolQC"].fillna("None")
    data["Fence"] = data["Fence"].fillna("None")
    data["MiscFeature"] = data["MiscFeature"].fillna("None")
    data["FireplaceQu"] = data["FireplaceQu"].fillna("None")


    data["LotFrontage"] = data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
    data["MasVnrArea"] = data.groupby("Neighborhood")["MasVnrArea"].transform(lambda x: x.fillna(x.median()))

    

    # FEATURE ENGINEERING
    
    
    data["GarageAge"] = data["YrSold"] - data["GarageYrBlt"]
    data["GarageAge"] = data["GarageAge"].fillna(0)
    data["GarageAge"] = data["GarageAge"].apply(lambda x: x if x >= 0 else 0)

    #! 
    data["GarageExists"] = data["GarageYrBlt"].notna().astype(int)
    #!
    
    data["HouseAge"] = data["YrSold"] - data["YearBuilt"] 
    data["HouseAge"] = data["HouseAge"].fillna(0)
    data["HouseAge"] = data["HouseAge"].apply(lambda x: x if x >= 0 else 0)


    data["TotalLivingArea"] = data["GrLivArea"] + data["TotalBsmtSF"]


    # Dropping unnecessary columns.
    data.drop("GarageYrBlt",axis=1,inplace=True)

    data.drop("YearBuilt",axis=1,inplace=True)

    # Mapping
    quality_map = {"None" : 0,
                   "Po": 1,
                   "Fa": 2,
                   "TA": 3,
                   "Gd": 4,
                   "Ex" : 5}


    for col in ordinal_quality_columns:

        data[col] = data[col].map(quality_map)
    
    data = pd.get_dummies(data, columns=nominal_cols, drop_first=True)

    return data

data = clean_data(data)



y = data.SalePrice
X = data.drop("SalePrice",axis = 1)

X_train, X_val, y_train, y_val = train_test_split(X,y,train_size=0.8,test_size=0.2,random_state=0)

print(X_train.dtypes[X_train.dtypes == "object"])

model = RandomForestRegressor(n_estimators=100,random_state=0)



pipeline = Pipeline(steps=[("model", model)])

pipeline.fit(X_train,y_train)

prediction = pipeline.predict(X_val)

result = mean_absolute_error(y_val,prediction)

print(result)





