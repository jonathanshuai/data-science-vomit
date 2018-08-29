# Show null values in columns
# Example output:
# incident_url                        0
# incident_url_fields_missing         0
# incident_characteristics          327
# source_url                        468
# sources                           610

null_count = df.isnull().sum()
null_count.sort_values()  # Ordered
