import pandas as pd

def main(df, img_size=512, y_small = 100, y_tiny = 50):

    df['shrink'] = df[['width', 'height']].max(axis=1) / img_size
    df['y_scaled'] = (df['ymax'] - df['ymin']) / df['shrink']
    df_l = df.loc[df.y_scaled > y_small, ['filename']].drop_duplicates()
    df_l['dummy'] = 1
    df = df.merge(df_l)
    df = df.loc[(df.dummy == 1) & (df.y_scaled > y_tiny),
                ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']]
    return df