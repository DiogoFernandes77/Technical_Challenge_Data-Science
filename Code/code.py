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

coef_social = 0.26
coef_ambiental = 0.45
coef_economico = 0.48


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
        #print(country_dict)
        if (user["Continent"] in cont_dict.keys()):
            cont_dict[user["Continent"]] = [ cont_dict[user["Continent"]][0]  + user["People"], cont_dict[user["Continent"]][1] + user["Planet"], cont_dict[user["Continent"]][2] + user["Profit"], cont_dict[user["Continent"]][3] + 1]
        else:
            cont_dict[user["Continent"]] = [user["People"], user["Planet"], user["Profit"], 1]
         
    reader.close()
    cont_info={}
    
    #estimativas dos modelos ML
    for continent in cont_dict.keys():
        media = [cont_dict[continent][0]/ cont_dict[continent][3] , cont_dict[continent][1]/ cont_dict[continent][3] , cont_dict[continent][2]/ cont_dict[continent][3] ]
        
        results = []
        for regressor in reg_list:
            #pred = regressor.predict(sc.transform([media]))
            pred = regressor.predict([media])
            results.append(pred[0])
        
        
        cont_info[continent] =  media + results
        
    #print(country_info)
    #Calculo dos ranks
    for x in range(3,8):#calcular o rank para cada algoritmo
        tmp = {}
        for continent in cont_info.keys():
            tmp[continent] = cont_info[continent][x]
        #print(tmp)
        ordered_tmp = dict(sorted(tmp.items(), key=lambda x:x[1]))
        #print(ordered_tmp)
        rank = 1
        for continent in ordered_tmp.keys():
            copy = cont_info[continent].copy()
            copy[x] = rank
            cont_info[continent] = copy
            rank += 1

    #Calculo do fator mais importante
    for continent in cont_info.keys():
        ft_s = cont_info[continent][0] * coef_social
        ft_a = cont_info[continent][1] * coef_ambiental
        ft_e = cont_info[continent][2] * coef_economico

        if(ft_s < ft_a and ft_s < ft_e):
            cont_info[continent].append("fator social")
        elif(ft_a < ft_s and ft_a < ft_e):
            cont_info[continent].append("fator ambiental")
        elif(ft_e < ft_s and ft_e < ft_a):
            cont_info[continent].append("fator economico")
        else:
            cont_info[continent].append("fatores empatados")
    
    ordered_dic = dict(sorted(cont_info.items(), key=lambda x:x[1][3]))
    #print(ordered_dic)
    
    schema = avro.schema.parse(open(current_dir + "/indiceContinente.avsc", "rb").read())
    writer = DataFileWriter(open("indiceContinente.avro", "wb"), DatumWriter(), schema)
    
    for continente in ordered_dic:
        tmp = ordered_dic[continente]
        writer.append({"Continente": continente, "Fator Social": int(tmp[0]), "Fator Ambiental": int(tmp[1]), "Fator Economico": int(tmp[2]), "Rank R.Linear": tmp[3], "Rank ElasticNet" : tmp[4], "Rank LassoLars": tmp[5], "Rank Lasso" : tmp[6], "Rank Ridge": tmp[7], "Fator Mais Importante": tmp[8] })
    writer.close()  
    # reader = DataFileReader(open(current_dir + "/indiceContinente.avro", "rb"), DatumReader())
    # for x in reader:
    #     print(x) 
    
    

def indicePais():
    
    reader = DataFileReader(open(current_dir + "/GreenCities.avro", "rb"), DatumReader())
    country_dict = {}
    for user in reader:
        #print(country_dict)
        if (user["Country"] in country_dict.keys()):
            country_dict[user["Country"]] = [ country_dict[user["Country"]][0]  + user["People"], country_dict[user["Country"]][1] + user["Planet"], country_dict[user["Country"]][2] + user["Profit"], country_dict[user["Country"]][3] + 1]
        else:
            country_dict[user["Country"]] = [user["People"], user["Planet"], user["Profit"], 1]
         
    reader.close()
    #print(country_dict)
    country_info={}
    
    #estimativas dos modelos ML
    for country in country_dict.keys():
        media = [country_dict[country][0]/ country_dict[country][3] , country_dict[country][1]/ country_dict[country][3] , country_dict[country][2]/ country_dict[country][3] ]
        
        results = []
        for regressor in reg_list:
            #pred = regressor.predict(sc.transform([media]))
            pred = regressor.predict([media])
            results.append(pred[0])
        
        
        country_info[country] =  media + results
        
    #print(country_info)
    #Calculo dos ranks
    for x in range(3,8):#calcular o rank para cada algoritmo
        tmp = {}
        for country in country_info.keys():
            tmp[country] = country_info[country][x]
        #print(tmp)
        ordered_tmp = dict(sorted(tmp.items(), key=lambda x:x[1]))
        #print(ordered_tmp)
        rank = 1
        for country in ordered_tmp.keys():
            copy = country_info[country].copy()
            copy[x] = rank
            country_info[country] = copy
            rank += 1

    #Calculo do fator mais importante
    for country in country_info.keys():
        ft_s = country_info[country][0] * coef_social
        ft_a = country_info[country][1] * coef_ambiental
        ft_e = country_info[country][2] * coef_economico

        if(ft_s < ft_a and ft_s < ft_e):
            country_info[country].append("fator social")
        elif(ft_a < ft_s and ft_a < ft_e):
            country_info[country].append("fator ambiental")
        elif(ft_e < ft_s and ft_e < ft_a):
            country_info[country].append("fator economico")
        else:
            country_info[country].append("fatores empatados")
    
    ordered_dic = dict(sorted(country_info.items(), key=lambda x:x[1][3]))
    schema = avro.schema.parse(open(current_dir + "/indicePais.avsc", "rb").read())
    writer = DataFileWriter(open("indicePais.avro", "wb"), DatumWriter(), schema)
    
    for country in ordered_dic:
        tmp = ordered_dic[country]
        writer.append({"Pais": country, "Fator Social": int(tmp[0]), "Fator Ambiental": int(tmp[1]), "Fator Economico": int(tmp[2]), "Rank R.Linear": tmp[3], "Rank ElasticNet" : tmp[4], "Rank LassoLars": tmp[5], "Rank Lasso" : tmp[6], "Rank Ridge": tmp[7], "Fator Mais Importante": tmp[8] })
    writer.close()  
    # reader = DataFileReader(open(current_dir + "/indicePais.avro", "rb"), DatumReader())
    # for x in reader:
    #     print(x) 
    

def fitModels():
    global reg_list, sc
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
    train_input = vec.fit_transform(x_train).toarray()
    test_input = vec.fit_transform(x_test).toarray()
    
    # sc = StandardScaler()
    # train_input = sc.fit_transform(x_train)
    # test_input = sc.transform(x_test)
    
   
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
        print("Intercept: \n", regressor.intercept_)
        # The mean squared error
        print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
        # The coefficient of determination: 1 is perfect prediction
        print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))
        
        print()
    
    
    
        

if __name__ == "__main__":
    csvToAvro()
    
    fitModels()
    
    indicePais()
    indiceContinente()
    
    
