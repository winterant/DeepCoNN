import argparse
import os
import sys
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.tokenize import WordPunctTokenizer

os.chdir(sys.path[0])


def process_dataset(json_path, select_cols, train_rate, csv_path):
    print('#### Read the json file...')
    if json_path.endswith('gz'):
        df = pd.read_json(json_path, lines=True, compression='gzip')
    else:
        df = pd.read_json(json_path, lines=True)
    df = df[select_cols]
    df.columns = ['userID', 'itemID', 'review', 'rating']  # Rename above columns for convenience
    # map user(or item) to number
    df['userID'] = df.groupby(df['userID']).ngroup()
    df['itemID'] = df.groupby(df['itemID']).ngroup()

    with open('stopwords.txt') as f:  # stop words
        stop_words = set(f.read().splitlines())
    with open('punctuations.txt') as f:  # useless punctuations
        punctuations = set(f.read().splitlines())

    def clean_review(review):  # clean a review using stop words and useless punctuations
        review = review.lower()
        for p in punctuations:
            review = review.replace(p, ' ')  # replace punctuations by space
        review = WordPunctTokenizer().tokenize(review)  # split words
        review = [word for word in review if word not in stop_words]  # remove stop words
        # review = [nltk.WordNetLemmatizer().lemmatize(word) for word in review]  # extract root of word
        return ' '.join(review)

    df = df.drop(df[[not isinstance(x, str) or len(x) == 0 for x in df['review']]].index)  # erase null reviews
    df['review'] = df['review'].apply(clean_review)

    train, valid = train_test_split(df, test_size=1 - train_rate, random_state=3)  # split dataset including random
    valid, test = train_test_split(valid, test_size=0.5, random_state=4)
    os.makedirs(csv_path, exist_ok=True)
    train.to_csv(os.path.join(csv_path, 'train.csv'), index=False, header=False)
    valid.to_csv(os.path.join(csv_path, 'valid.csv'), index=False, header=False)
    test.to_csv(os.path.join(csv_path, 'test.csv'), index=False, header=False)
    print(f'#### Split and saved dataset as csv: train {len(train)}, valid {len(valid)}, test {len(test)}')
    print(f'#### Total: {len(df)} reviews, {len(df.groupby("userID"))} users, {len(df.groupby("itemID"))} items.')
    return train, valid, test


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', dest='data_path',
                        default='Digital_Music_5.json.gz',
                        help='Selected columns of above dataset in json format.')
    parser.add_argument('--select_cols', dest='select_cols', nargs='+',
                        default=['reviewerID', 'asin', 'reviewText', 'overall'])
    parser.add_argument('--train_rate', dest='train_rate', default=0.8)
    parser.add_argument('--save_dir', dest='save_dir', default='./music')
    args = parser.parse_args()

    start_time = time.perf_counter()
    process_dataset(args.data_path, args.select_cols, args.train_rate, args.save_dir)
    end_time = time.perf_counter()
    print(f'## preprocess.py: Data loading complete! Time used {end_time - start_time:.0f} seconds.')
