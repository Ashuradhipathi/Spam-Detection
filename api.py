from serpapi import GoogleSearch

params = {
  "q": "Coffee",
  "location": "Austin, Texas, United States",
  "hl": "en",
  "gl": "us",
  "google_domain": "google.com",
  "api_key": "9a4a56b00179906512733cd9492838cf4f65d04459c2739121a9a716d1b85a3b"
}

search = GoogleSearch(params)
results = search.get_dict()