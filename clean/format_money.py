# Format money: basically removes commas and $

format_balance = lambda b: float(b.replace(',', '').strip('$'))
df['Balance'] = df['Balance'].apply(format_balance)