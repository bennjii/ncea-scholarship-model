import pandas as pd 
import csv

#data["Indexes"] = data["full_road_name"].str.find(sub) 

#print(data["full_road_name"].str.find(sub, start))

#print(df.loc[df['full_road_name'] == sub])


# reading csv file from url  
df = pd.read_csv("./data/test/remuera/adresses.csv") 
   
# dropping null value columns to avoid errors 
df.dropna(inplace = True) 

sub ='Kenny Road'
dub = '1'

save_line = 0;

with open("./data/test/remuera/adresses.csv", "r") as f:
    reader = csv.reader(f)
    for line_num, content in enumerate(reader):
        if content[13] == sub:
            if content[12] == dub:
                print("FOUND! LINE", line_num + 1, " : ", content[16], content[17]) # , "\n", content
                save_line = line_num+1

subdub = dub.upper() + " " + sub.upper()
print(subdub)