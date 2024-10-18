import pandas as pd

#Import data

data1 = pd.read_csv("culera-tweets")
data1 = data1[['content']]
print(len(data1))
data2 = pd.read_csv("new-puta-tweets")
data2 = data2[['content']]
print(len(data2))
data3 = pd.read_csv("almostcompletedset")
data3 = data3[['content']]
#data4 = pd.read_csv("idiota-imbecil-tweets")
#data5 = pd.read_csv("maldita-tweets")
#data6 = pd.read_csv("mierda-es-tweets")
#data7 = pd.read_csv("tweets")
#data8 = pd.read_csv("verga-tweets")
#data9 = pd.read_csv("zorra-tweets")


dataset = pd.concat([data1,data2, data3])
dataset = dataset[['content']]

dataset = dataset[['content']].astype(str)



#Dropping ALL duplicate values
dataset.drop_duplicates()
#Dropping ALL null values
dataset = dataset.dropna()

#Print data length
print("length of dataframe: ", len(dataset))

dataset.to_csv("profanitydataset")
