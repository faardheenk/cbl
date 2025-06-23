from fuzzywuzzy import fuzz

def main():
    # str1 = "Royal Car Rentle LTD"
    # str2 = "Royal Car Rental LTD MCB Holding BANK COMMERCIAL LTD"

    # str1 = "*ROYAL CAR RENTAL LTD*"
    # str2 = "*NADS'S CAR RENTAL LTD*"

    
    str1 = "SIKA (MAURITIUS) LTD"
    str2 = "*CERIDIAN (MAURITIUS) LTD*"
    
    print(f"Simple ratio: {fuzz.ratio(str1, str2)}")
    print(f"Partial ratio: {fuzz.partial_ratio(str1, str2)}")
    print(f"Token sort ratio: {fuzz.token_sort_ratio(str1, str2)}")
    print(f"Token set ratio: {fuzz.token_set_ratio(str1, str2)}")

if __name__ == "__main__":
    main() 