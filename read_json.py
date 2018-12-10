import json

if __name__ == "__main__":
    j = 0
    lines = []
    with open("data.json", "r") as f:
        for line in f:
            lines.append(line)
            j += 1
            if j > 1000:
                break
    with open("data1.json", "w") as f1:
        f1.writelines(lines)


    #with open('test.json') as f2:
    #    json_dic = json.load(f2)