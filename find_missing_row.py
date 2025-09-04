import pandas as pd

# Read the sheets
partial_cbl = pd.read_excel('data/RESULT.xlsx', sheet_name='partial match cbl')
partial_combined = pd.read_excel('data/RESULT.xlsx', sheet_name='Partial Matches Combined')

print(f'Partial Match CBL records: {len(partial_cbl)}')
print(f'Partial Matches Combined records: {len(partial_combined)}')

print('\nChecking for missing CBL records...')
missing_records = []

for idx, cbl_row in partial_cbl.iterrows():
    cbl_name = str(cbl_row['ClientName']).strip()
    cbl_amount = cbl_row['Amount_Clean']
    found = False
    
    for _, combined_row in partial_combined.iterrows():
        combined_name = str(combined_row['ClientName']).strip()
        combined_amount = combined_row['Amount_Clean']
        
        if cbl_name == combined_name and abs(cbl_amount - combined_amount) < 0.01:
            found = True
            break
    
    if not found:
        missing_records.append({
            'index': idx, 
            'ClientName': cbl_name, 
            'Amount': cbl_amount
        })
        print(f'Missing record found: Index {idx}, Client: {cbl_name}, Amount: {cbl_amount}')

print(f'\nTotal missing records: {len(missing_records)}')

# Also check the reverse - see if there are any extra records in combined
print('\nChecking for extra records in combined sheet...')
extra_records = []

for idx, combined_row in partial_combined.iterrows():
    combined_name = str(combined_row['ClientName']).strip()
    combined_amount = combined_row['Amount_Clean']
    found = False
    
    for _, cbl_row in partial_cbl.iterrows():
        cbl_name = str(cbl_row['ClientName']).strip()
        cbl_amount = cbl_row['Amount_Clean']
        
        if combined_name == cbl_name and abs(combined_amount - cbl_amount) < 0.01:
            found = True
            break
    
    if not found:
        extra_records.append({
            'index': idx, 
            'ClientName': combined_name, 
            'Amount': combined_amount
        })
        print(f'Extra record found: Index {idx}, Client: {combined_name}, Amount: {combined_amount}')

print(f'\nTotal extra records: {len(extra_records)}')

# Show some sample records from both sheets for comparison
print('\nSample records from Partial Match CBL:')
print(partial_cbl[['ClientName', 'Amount_Clean']].head(10))

print('\nSample records from Partial Matches Combined:')
print(partial_combined[['ClientName', 'Amount_Clean']].head(10))
