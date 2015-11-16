import pymysql
db = pymysql.connect("localhost","root","PASSWD","Android")
cursor = db.cursor()
import matplotlib.pyplot as plt
cursor.execute("SELECT ViewCount FROM Posts")

result = cursor.fetchall()
l1 = []
for i in range(0,len(result)) :

	l1.append(result[i][0])

l2 = sorted(l1,reverse = True)
l3 = []
for i in range(len(l2)) :
	l3.append(i)

l4 = list(set(l2))
xmin = 0
xmax = 80000
ymin = 0
ymax = 550000
axes = plt.gca()
axes.set_xlim([xmin,xmax])
axes.set_ylim([ymin,ymax])

plt.plot(l1,'-b',linewidth = 3,label = "post views")
plt.legend(fontsize = 21)#, weight = 'bold')
plt.xticks(fontsize = 21, fontweight = 'semibold')
plt.yticks(fontsize = 21,fontweight = 'semibold')
plt.xlabel("Post Id", fontsize = 21,fontweight = 'semibold',fontname = 'Times New Roman')
plt.ylabel("Post Id", fontsize = 21,fontweight = 'semibold')
plt.show()
