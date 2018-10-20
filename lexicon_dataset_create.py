"""
This script is used to convert the frequency dataset into a normalised dataset

Issues: 
    -- BUG: dataset must be opened in excel and the first column deleted as it is blank
    -- FIXME: make the try except catch the specific exception
"""
import csv
from decorators import timer

def read_data(file_path: str, delim: str = ',') -> list:
    full_data = []
    with open(file_path) as data_file:
        reader = csv.reader(data_file, delimiter=delim)
        for row in reader:
            full_data.append(row)

    return full_data

def write_data(file_path: str, height: int, width: int, Matrix: list, delim: str = ','):
    with open(file_path, 'w') as file_obj:
        for x in range(0, height):
            line = ""
            for y in range(0, (width - 1)):
                line = f"{line}{delim}{Matrix[x][y]}"
            file_obj.write(f"{line}\n")

@timer
def edit_lexicon(width: int, height: int, data_main: list, lexicon: list) -> list:
    # dataset copy for manipulating
    Matrix = [[0 for x in range(width)] for y in range(height)]

    for x in range(0, (width - 1)):
        feature_name = data_main[0][x]
        for lexicon_instance in lexicon:
            # check if the feature name is found within the lexicon (e.g. "game" has a weighting in the lexicon)
            if feature_name in lexicon_instance:
                Matrix = edit_instance(height, x, feature_name, data_main, lexicon_instance, Matrix)
                # once word is found move onto the next one
                break
            else:
                for y in range(0, height):
                    Matrix[y][x] = 0
            
    # overwrite the feature names to the matrix for cases like "0" as a feature name where it would have been multiplied above
    for x in range(0, (width - 1)):
        feature_name = data_main[0][x]
        Matrix[0][x] = feature_name
    
    return Matrix

def edit_instance(height: int, x: int, feature_name: str, data_main: list, lexicon_instance: list, Matrix: list) -> list:
    for y in range(0, height):
        # try to multiply the current instances feature by the weighting from the lexicon and overwrite the matrix
        try:
            new_cell = float(data_main[y][x]) * float(lexicon_instance[1])
            Matrix[y][x] = new_cell
        # if the above failed then it is the features string name so it is just overwritten
        # FIXME: add specific error handling
        except:
            Matrix[y][x] = feature_name

    return Matrix

if __name__ == "__main__":
    lex_path = "C:/Users/Erik/Documents/LocalSandbox/Sentiment-Analysis/Review_Dataset/Games_senti_lexicon.tsv"
    review_path = "C:/Users/Erik/Documents/LocalSandbox/Sentiment-Analysis/Review_Dataset/reviews_Video_Games_train.csv"
    out_path = "C:/Users/Erik/Documents/LocalSandbox/Sentiment-Analysis/Review_Dataset/lexicon_dataset_train.csv"

    lexicon = read_data(lex_path, delim='\t')
    reviews = read_data(review_path)    # default delimiter is comma so don't need to set as the review file is .csv

    width = 2001
    height = 5000

    Matrix = edit_lexicon(width, height, reviews, lexicon)
    write_data(out_path, height, width, Matrix)