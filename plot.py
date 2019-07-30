import csv
import matplotlib.pyplot as plt

if __name__=='__main__':
	with open("out.csv", "r", newline="") as f:
		reader=csv.reader(f)
		result=[[float(v) for v in r] for r in reader]
		result.reverse()

		plt.imshow(result, cmap="gist_rainbow")
		plt.gca().invert_yaxis()
		plt.draw()

		pp=plt.colorbar(orientation="vertical")
		pp.set_label("voltage", fontsize=24)
		plt.show()
