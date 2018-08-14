import csv

def get_class_name():
    f = open('/media/zhonghao/New Volume/openimage/annotations/class-descriptions-boxable.csv', 'r')
    csvreader = csv.reader(f, delimiter=',')
    classname = ['\n']
    for line in csvreader:
        classname.append(line[1])
    f.close()
    return classname

def get_class_symbol():
    f = open('/media/zhonghao/New Volume/openimage/annotations/class-descriptions-boxable.csv', 'r')
    csvreader = csv.reader(f, delimiter=',')
    classname = ['\n']
    for line in csvreader:
        classname.append(line[0])
    f.close()
    return classname