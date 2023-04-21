import tomli as tomllib

with open("test/test.toml", "rb") as f:
    test = tomllib.load(f)

print(test)
