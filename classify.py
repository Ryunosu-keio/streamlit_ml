import pandas as pd
import csv

path = "input/all_riku_one.csv"
df = pd.read_csv(path)


cgy_list = []
for index, data in df.iterrows():
    if data["back car"] <= 1.5 :
        if data["section"] <= 4.75:
            if data["online"] <= 0.5:
                if data["num of add alart"] <= 0.5:
                    cgy = "1"
                else:
                    cgy = "2"
            else:
                if data["num of add alart"] <= 0.5:
                    cgy = "3"
                else:
                    cgy = "4"
        else:
            if data["num of add alart"] <= 2.5:
                if data["num of add alart"] <=1.5:
                    cgy = "5"
                else:
                    cgy = "6"
            else:
                cgy = "7"
    else:
        if data["num of add alart"] <= 1.5:
            if data["adjacent car"] <= 1.5:
                if data["small car add"] <= 1.5:
                    cgy = "8"
                else:
                    cgy = "9"
            else:
                if data["section"] <= 1.75:
                    cgy = "10"
                else:
                    cgy = "11"
        else:
            if data["section"] <= 4.25:
                if data["section"] <= 1.25:
                    cgy = "12"
                else:
                    cgy = "13"
            else:
                if data["section"] <= 4.75:
                    cgy = "14"
                else:
                    cgy = "15"
    cgy_list.append([cgy])
print(cgy_list)

with open("output/classify_list.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(cgy_list)
