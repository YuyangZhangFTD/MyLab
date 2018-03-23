import numpy as np


def generate_document(doc_num):
    with open("input/docs.csv", "w") as w:
        threshold = 0.9
        for _ in range(doc_num):
            doc = []
            for _ in range(np.random.randint(150, 200)):
                if np.random.rand() < threshold:
                    doc.append(str(np.random.randint(0, 40)))
                else:
                    doc.append(str(np.random.randint(40, 50)))
            w.write(",".join(doc) + "\n")


if __name__ == '__main__':
    # generate_document(50)
    print("hello world")
