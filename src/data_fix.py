import pandas as pd

# 1. Load your current data
df = pd.read_csv("./data/train.csv")
print(f"Null Captions before fix: {df['caption'].isna().sum()}")

# 2. Function to fill missing caption from content
def smart_fill(row):
    # Check if caption is null or the string 'nan'
    if pd.isna(row['caption']) or str(row['caption']).lower() == 'nan':
        # Grab the first 5 words of the content as a temporary 'Topic'
        content_words = str(row['content']).split()
        # Create a short topic string (e.g., "Thalcave's people faded as...")
        new_topic = " ".join(content_words[:5]) + "..."
        return new_topic
    return row['caption']

# 3. Apply the fix
df['caption'] = df.apply(smart_fill, axis=1)

# 4. Save it back
df.to_csv("./data/train.csv", index=False)
print(f"Null Captions after fix: {df['caption'].isna().sum()}")
print("Done! Your data is now 100% enriched.")