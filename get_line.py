import sys

file = sys.argv[1]

l = []
for line in open(file):
    line = line.strip()
    l.append(line)
    # print(line)
print(len(l))