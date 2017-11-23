from NB import NBC
import prepro
if __name__ == "__main__":
    pro = prepro.preprocess(['f1.csv','f2.csv','f3.csv','f4.csv','f5.csv'],['CONTENT','CLASS'])
    data = pro.read_and_clean()
    clf = NBC(data)
    clf.feature_extraction()
    clf.train()
    while True:
        s = input()
        if clf.predict(s)==1:
            print("Its spam! Careful")
        else:
            print("Its not-spam. You are safe.")
