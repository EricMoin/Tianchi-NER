import json

data_json = []
with open("data/dev.json", "r", encoding="utf-8") as file:
    for line in file:
        data_json.append(json.loads(line))

with open('data/extra_dev_2.txt', "w", encoding="utf-8") as file:
    for data in data_json:
        file.write(data['sentence1'] + "\n")
        file.write(data['sentence2'] + "\n")
