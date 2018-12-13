import json

if __name__ == "__main__":
#    j = 0
#    lines = []
#    with open("programs_training.json", "r") as f:
#        for line in f:
#            lines.append(line)
#            j += 1
#            if j > 100:
#                break
#    with open("data1.json", "w") as f1:
#        f1.writelines(lines)

    k = 0
    with open('data1.json') as f2:
        for line in f2:
            data = json.loads(line)
            print(json.dumps(data, indent=2))
            k += 1
            if k > 1:
                break

