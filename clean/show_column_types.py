# Create a DataFrame with all the column names and types

type_df = pd.DataFrame()
type_df['column'] = df.columns
type_df['type'] = [df[c].dtype for c in df.columns]
type_df