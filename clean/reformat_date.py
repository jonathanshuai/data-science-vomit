from datetime import datetime
import dateutil

# Method 1: Explicit parsing
format_date = lambda d: datetime.strptime(d, "%m/%d/%Y").strftime('%Y-%m-%d')
df['birthday'] = df[df['birthday'].notnull()]['birthday'].apply(format_date)

# Method 2: Parsing using dateutil (probably should double check this, but it's easier)

format_date = lambda d: dateutil.parser.parse(d).strftime('%Y-%m-%d')
df['birthday'].apply(format_date)


# Finally, the easiest:
pd.to_datetime(df['birthday'])
