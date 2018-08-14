import pdb
import csv
'''
i = 0
f = open('outputs.csv', 'r')
f_write = open('submit.csv', 'w')
for line in f:
    f_write.write(line)
f.close()
f = open('outputs2.csv', 'r')
for line in f:
    f_write.write(line)
f.close()
f_write.close()
'''
'''
thre = 0.6
f = open('submit.csv', 'r')
f_write = open('submit2.csv', 'w')
#csvwriter = csv.writer(f_write, delimiter=' ')
f_write.write('ImageId,PredictionString\n')
i = 0
j = 0
for line in f:
    if i == 0:
        i = 1
        continue
    j += 1
    line_split = line.split(',')
    if len(line_split) == 1:
        f_write.write(line)
        continue

    name = line_split[0]
    content = line_split[1].split(' ')
    con_write = []
    for k in range(len(content)/6):
        if float(content[k*6+1]) > thre:
            temp = [content[k*6], content[k*6+1]]
            for w in range(2, 6):
                if float(content[k*6+w]) > 1.0:
                    temp.append(str('1.0'))
                else:
                    temp.append(str(float(content[k*6+w])))
            for item in temp:
                con_write.append(item)
    f_write.write(name+','+' '.join(con_write)+'\n')


f.close()
f_write.close()
print(j)
'''
f = open('./submit2.csv', 'r')
i = 0
j = 0
for line in f:
    i += len(line.split(' ')) / 6.
    j += 1
print(j)
print(i/100000)
#print(i)
