import csv

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#This script is used to convert the frequency dataset into a proper 
#PLEASE NOTE, THERE IS A BUG THAT REQUIRES YOU TO OPEN THE DATASET IN EXCEL AND DELETE THE FIRST COLUMN AS IT IS BLANK
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#read the lexicon file into a list
lexicon = []
with open("D:/IS/lexicon.tsv") as tsvfile:
  reader = csv.reader(tsvfile, delimiter='\t')
  for row in reader:
    lexicon.append(row)


#read the frequency dataset into a list
reviews = []
with open("D:/IS/csv/reviews_Video_Games_test.csv") as tsvfile:
  reader = csv.reader(tsvfile, delimiter=',')
  for row in reader:
      reviews.append(row)


#width and height of the dataset (2001 features by 5000 instances)
counter = 0
w,h = 2001, 5000;
#create a blank dataset the size of the original
Matrix = [[0 for x in range(w)] for y in range(h)]


#iterate through the features
for x in range(0,2000):
    #get feature name
    feauture_name = reviews[0][x]
    #iterate through each word in the lexicon
    for lexicon_instance in lexicon:
        #check if the feature name is found within the lexicon (e.g. "game" has a weighting in the lexicon)
        if feauture_name in lexicon_instance:
            #iterate through the instances of that feature
            for y in range(0, 5000):
                #try to multiply the current instances feature by the weighting from the lexicon and overwrite the matrix
                try:
                    new_cell = float(reviews[y][x]) * float(lexicon_instance[1])
                    Matrix[y][x] = new_cell
                #If the above failed then it is the features string name so it is just overwritten
                except:
                    Matrix[y][x] = feauture_name
            #once the word is found, stop looking through the lexicon and move onto the next feature
            break
        else:
            #set weighting of that feature to 0
            for y in range(0, 5000):
                Matrix[y][x] = 0
            pass


#overwrite the feature names to the matrix for cases like "0" as a feature name where it would have been multiplied above
for x in range(0,2000):
    feauture_name = reviews[0][x]
    Matrix[0][x] = feauture_name


#create the new dataset .csv file
file_obj = open("D:/IS/lex_ds/lexicon_dataset_train.csv", "w")
#iterate through the instances first
for x in range(0,5000):
    #create a line string whihch will store all values of that instance
    line = ""
    #iterate through each instance
    for y in range(0,2000):
        #append a comma and the next value in the csv file
        line = line + "," + str(Matrix[x][y])
    #write that instance to the file with a new line after
    file_obj.write(line + "\n")
#close the dataset file
file_obj.close()
