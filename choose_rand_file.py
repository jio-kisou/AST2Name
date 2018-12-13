import random

if __name__ == "__main__":
    lines = []
    with open("eval_list.txt", "r") as f:
        for line in f:
            l = [0, 1, 2]
            k = random.choice(l)
            if k == 0:
                lines.append(line)

    with open("eval_list_1of3.txt", "w") as f1:
        f1.writelines(lines)