import re

line_limit = 1000

def refactoring_data():
    data_path = "assets/human_text.txt"
    data_path2 = "assets/robot_text.txt"
    # Defining lines as a list of each line
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
    with open(data_path2, 'r', encoding='utf-8') as f:
        lines2 = f.read().split('\n')
    lines = [re.sub(r"\[\w+\]",'hi',line) for line in lines[:line_limit]]
    lines = [" ".join(re.findall(r"\w+",line)) for line in lines]
    lines2 = [re.sub(r"\[\w+\]",'',line) for line in lines2]
    lines2 = [" ".join(re.findall(r"\w+",line)) for line in lines2[:line_limit]]
    # grouping lines by response pair
    pairs = list(zip(lines,lines2))
    #random.shuffle(pairs)
    with open ('assets/human-robot.txt', 'w+') as f:
        for line in pairs:
            f.writelines(line[0] + '\t' + line[1] + '\n')


if __name__ == "__main__":
    refactoring_data()