# Practical 2: Use regex patterns to extract usernames from email addresses,
# hashtags, dates, and phone numbers present in a given text.
#
# HOW TO RUN:
# pip install nltk gutenberg  (only if using gutenberg dataset)
# python practical2.py

import re

# Sample text containing emails, hashtags, dates, and phone numbers
text = """
Contact us at rohan21@example.com or new_jane.doe123@mail.co for help.
Follow us on #NLP #ArtificialIntelligence #Python.
The event is scheduled on 29-03-2024 and also on 02-01-2002.
Call us at +919978452033 or reach out at 9876543212.
Also write to microsoft@mycompany.net and new_test_account@test.info
"""

# Extract usernames from email addresses
usernames = re.findall(r'([\w.]+)@\w+\.\w+', text)
print("Usernames from Emails:")
print(usernames)

# Extract hashtags
hashtags = re.findall(r'#\w+', text)
print("\nHashtags:")
print(hashtags)

# Extract dates (DD-MM-YYYY format)
dates = re.findall(r'\b\d{2}-\d{2}-\d{4}\b', text)
print("\nDates:")
print(dates)

# Extract phone numbers (10-digit or with +91 country code)
phone_numbers = re.findall(r'\+?\d{10,13}', text)
print("\nPhone Numbers:")
print(phone_numbers)
