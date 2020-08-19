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
                    print("SEARCH RESULT FOUND! LINE", line_num + 1, " : ", content[16], content[17]) # , "\n", content
                    return content

def searchArray():
    with open("./data/test/remuera/sales.csv", "r") as f, \
        open('./data/test/remuera/output_1.csv', 'w', newline='') as write_obj:

        reader_ = csv.reader(f)
        writer_ = csv.writer(write_obj)

        writer_.writerow(["address","owners","suburb","town","ta_name","property_type","sale_date","capital_value","gross_sale_price","bedrooms_min","land_area","floor_area","building_age","listing_date","provisional_sale_price","provisional_sale_date","","","long","lat"])

        enum = 0
        counter = 0

        for line_num, content in enumerate(reader_, start=1):
            if(enum > 0):
                enum += 1
                subdub = content[0].split(" ", 1)

                sub = subdub[0].upper()
                dub = subdub[1].upper()
                
                print("SEARCHING FOR ", sub, dub)
                result = findInAdressBook(sub, dub)

                if(result != None):
                    print(sub, dub)
                    print(result)
                    print("\n")
                    counter += 1

                    content.append(result[16])
                    content.append(result[17])
                    writer_.writerow(content)
                else:
                    print("FAILED, NO CORROSPRONDING MATCH WAS FOUND IN THE ADRESS BOOK")
            else:
                enum += 1

        print('TOTAL SEARCHES FOUND: ', counter)
        print('TOTAL SEARCHES MADE: ', enum)

searchArray()