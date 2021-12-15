import csv
import pandas as pd


def data_cleanup():
    # Cleaning up the data
    with open('lyrics.csv', 'r') as inp, open('lyrics_out.csv', 'w') as out:
        writer = csv.writer(out)
        for row in csv.reader(inp):
            if row[4] != "Alkebulan" and row[4] != "Other" and row[4] != "" and row[4] != "Not Available" and row[
                4] != "zora sourit" and row[5] != "":
                writer.writerow(row)


def multi_class_data():
    data_cleanup()

    df = pd.read_csv('lyrics_out.csv')

    # choose which genres to analyze
    # df = df[df['genre'] != "Pop"]
    df = df[df['genre'] != "Folk"]
    # df = df[df['genre'] != "Jazz"]
    df = df[df['genre'] != "R&B"]
    df = df[df['genre'] != "Indie"]
    df = df[df['genre'] != "Electronic"]
    df = df[df['genre'] != "Metal"]
    # df = df[df['genre'] != "Country"]
    # df = df[df['genre'] != "Hip-Hop"]
    # df = df[df['genre'] != "Rock"]

    # add a new column with word count of the lyrics of a song
    # remove rows with lyrics count less than 100
    df['word_count'] = df['lyrics'].str.split().str.len()
    df = df[df['word_count'] > 100]

    # make all lowercase so easier to analyze
    df["lyrics"] = df['lyrics'].str.lower()
    # get rid or irrelevant symbols
    df['lyrics'] = df['lyrics'].str.strip('[]')
    df['lyrics'] = df['lyrics'].str.strip('()')
    df["lyrics"] = df['lyrics'].str.replace('[^\w\s]', '')
    df["lyrics"] = df['lyrics'].str.replace('chorus', '')
    df["lyrics"] = df['lyrics'].str.replace(':', '')
    df["lyrics"] = df['lyrics'].str.replace(',', '')
    df["lyrics"] = df['lyrics'].str.replace('verse', '')
    df["lyrics"] = df['lyrics'].str.replace('x1', '')
    df["lyrics"] = df['lyrics'].str.replace('x2', '')
    df["lyrics"] = df['lyrics'].str.replace('x3', '')
    df["lyrics"] = df['lyrics'].str.replace('x4', '')
    df["lyrics"] = df['lyrics'].str.replace('x5', '')
    df["lyrics"] = df['lyrics'].str.replace('x6', '')
    df["lyrics"] = df['lyrics'].str.replace('x7', '')
    df["lyrics"] = df['lyrics'].str.replace('x8', '')
    df["lyrics"] = df['lyrics'].str.replace('x9', '')

    # put in order of year to make sorting easier later
    df.sort_values(by=['year'], inplace=True)

    # replace carriage returns
    df = df.replace({'\n': ' '}, regex=True)

    # remove punctuations
    df["lyrics"] = df['lyrics'].str.lower().replace('[^\w\s]', '')

    # create datasets old vs new
    new_df = df[df['year'] > 2003]
    old_df = df[df['year'] <= 2003]
    old_df = old_df[old_df['year'] > 1965]

    # group by genre
    old_df = old_df.groupby('genre').head(600)
    new_df = new_df.groupby('genre').head(600)

    # create files
    old_df.to_csv('lyrics_old.csv', index=False)
    new_df.to_csv('lyrics_new.csv', index=False)

    # optional: print how many songs are available for each genre in old section
    '''old_Pop = old_df[old_df['genre'] == 'Pop']
    old_Rock = old_df[old_df['genre'] == 'Rock']
    old_Jazz = old_df[old_df['genre'] == 'Jazz']
    old_HipHop = old_df[old_df['genre'] == 'Hip-Hop']
    old_Country = old_df[old_df['genre'] == 'Country']
    print(len(old_Pop))
    print(len(old_Jazz))
    print(len(old_Rock))
    print(len(old_HipHop))
    print(len(old_Country))'''

if __name__ == '__main__':
    multi_class_data()

