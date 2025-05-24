import json

data_json = []
with open("data/address.txt", "r", encoding="utf-8") as file:
    total_len = len(file)
    for i in range(30000):
        if i % 1000 == 0:
            print(f"Processing line {i} of {total_len}")
        line = file.readline()
        if line:
            data_json.append(line)
