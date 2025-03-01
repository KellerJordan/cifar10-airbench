class Test:
    def __enter__(self):
        pass
    def __exit__(self, type, value, traceback):
        print(5)

with Test():
    print(6)
