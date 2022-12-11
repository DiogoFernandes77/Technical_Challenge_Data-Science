import avro.schema
import csv
import os
from avro.datafile import DataFileReader, DataFileWriter
from avro.io import DatumReader, DatumWriter
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


current_dir = os.getcwd()



def csvToAvro():
    #schema
    schema = avro.schema.parse(open(current_dir + "/schema.avsc", "rb").read())
    #writer
    writer = DataFileWriter(open("GreenCities.avro", "wb"), DatumWriter(), schema)
    #Reading csv and create .avro db
    with open(current_dir + "/GreenCities-Data.csv") as f:
        csv_file = csv.reader(f)
        next(csv_file)
        for count, row in enumerate(csv_file):
            writer.append({"city": row[0], "People": int(row[1]), "Planet": int(row[2]), "Profit": int(row[3]), "Overall": int(row[4]), "Country": row[5], "Continent": row[6]})
        writer.close()        
    
def indiceContinente():
    reader = DataFileReader(open(current_dir + "/GreenCities.avro", "rb"), DatumReader())
    cont_dict = {}
    for user in reader:
        if (user["Continent"] in cont_dict.keys()):
            cont_dict[user["Continent"]] = cont_dict[user["Continent"]] + 1
        else:
            cont_dict.update({user["Continent"]: 1})
         
    reader.close()
    #sort dict
    cont_dict = dict(sorted(cont_dict.items(), key=lambda x:x[1],reverse=True))
    
    schema = avro.schema.parse(open(current_dir + "/indiceContinente.avsc", "rb").read())
    writer = DataFileWriter(open("indiceContinent.avro", "wb"), DatumWriter(), schema)
    cnt = 1
    for key in cont_dict:
        writer.append({"Continent": key, "Ocurrences": cont_dict[key], "Rank": cnt})
        cnt +=1
    writer.close()  
    
    #check results
    # reader = DataFileReader(open(current_dir + "/indiceContinent.avro", "rb"), DatumReader())
    # for x in reader:
    #     print(x) 
    
    

def indicePais():
    reader = DataFileReader(open(current_dir + "/GreenCities.avro", "rb"), DatumReader())
    country_dict = {}
    for user in reader:
        if (user["Country"] in country_dict.keys()):
            country_dict[user["Country"]] = country_dict[user["Country"]] + 1
        else:
            country_dict.update({user["Country"]: 1})
         
    reader.close()
    print(country_dict)
    return country_dict

def findFactor():
    reader = DataFileReader(open(current_dir + "/GreenCities.avro", "rb"), DatumReader())
    data_x = []
    data_y = []
    data_names = []
    for city in reader:
        city.pop("Country")
        city.pop("Continent")
        data_y.append(city.pop("Overall"))
        data_names.append(city.pop("city"))
        data_x.append(city)
    # print(data_x)
    # print(data_y)
    # print(data_names)
    x_train, x_test, y_train, y_test, names_train, names_test = train_test_split(data_x, data_y, data_names, test_size=0.2, random_state=2) 
    
    vec = DictVectorizer()
    x_train = vec.fit_transform(x_train).toarray()
    x_test = vec.fit_transform(x_test).toarray()
    
    sc = StandardScaler()
    train_input = sc.fit_transform(x_train)
    test_input = sc.transform(x_test)
    
    reg_list = []
    reg_list.append(linear_model.LinearRegression())
    reg_list.append(ElasticNet(random_state=0))
    reg_list.append(linear_model.LassoLars(alpha=0.01))
    reg_list.append(linear_model.Lasso(alpha=0.1))
    reg_list.append(linear_model.Ridge(alpha=.5))
    
    
    for regressor in reg_list:
        
        regressor.fit(train_input, y_train)
        y_pred = regressor.predict(test_input)
        
        #for x in range(len(y_pred)):
            #print(names_test[x] + " , " + str(y_test[x]) + " , " + str(y_pred[x]))
        print("Coefficients: \n", regressor.coef_)
        # The mean squared error
        print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
        # The coefficient of determination: 1 is perfect prediction
        print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))
        print()
    
    
    
        

if __name__ == "__main__":
    #csvToAvro()
    #reading avro db
    #indiceContinente()
    #indicePais()
    findFactor()
