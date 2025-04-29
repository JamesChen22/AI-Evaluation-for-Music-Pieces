import pandas as pd

# Load the two datasets
df1 = pd.read_csv('low_popularity_spotify_data.csv')
df2 = pd.read_csv('high_popularity_spotify_data.csv')

# Concatenate them along the rows
combined_df = pd.concat([df1, df2], ignore_index=True)

# Check the shape of the combined dataframe
print("Combined shape:", combined_df.shape)

combined_df.to_csv('review_dataset.csv', index=False)
