str = "Hello, My, name, is, Michael"

str.split(",")

cleaned = []
for s in str.split(","):
    cleaned.append(s.strip())


print(cleaned)