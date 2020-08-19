import pandas as pd 
import csv

#data["Indexes"] = data["full_road_name"].str.find(sub) 

#print(data["full_road_name"].str.find(sub, start))

#print(df.loc[df['full_road_name'] == sub])


# reading csv file from url  
df = pd.read_csv("./data/test/remuera/adresses.csv") 
   
# dropping null value columns to avoid errors 
df.dropna(inplace = True) 

sub ='Orakei Road'
dub = '5'

subdub = dub.upper() + " " + sub.upper()
#print(subdub)

save_line = 0

# SUB = TEXT
# DUB = NUMBER

def findInAdressBook(sub, dub):
    with open("./data/test/remuera/adresses.csv", "r") as f:
        reader = csv.reader(f)

        for line_num, content in enumerate(reader):
            #print(content[13].upper(), dub, "---------------------------------")
            if content[13].upper() == dub:
                if content[12].upper() == sub:
                    print("FOUND! LINE", line_num + 1, " : ", content[16], content[17]) # , "\n", content
                    return content

def searchArray():
    with open("./data/test/remuera/sales.csv", "r") as f:
        reader_ = csv.reader(f)
        enum = 0

        for line_num, content in enumerate(reader_, start=1):
            enum = enum + 1

            if(enum > 2):
                subdub = content[0].split(" ", 1)

                sub = subdub[0].upper()
                dub = subdub[1].upper()

                result = findInAdressBook(sub, dub)

                if(result != None):
                    print(sub, dub)
                    print(findInAdressBook(sub, dub))
                    print("\n")

searchArray()