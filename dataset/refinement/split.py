import splitfolders

if __name__ == "__main__":
    splitfolders.ratio("C:\\Users\\manic\\Documents\\woods-edge-correction\\dataset\\refinement\\input", output="output",
                       seed=1337, ratio=(.8, .2), group_prefix=None, move=False)
