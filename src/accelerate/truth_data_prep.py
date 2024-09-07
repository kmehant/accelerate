from datasets import load_dataset, Dataset
import random
import pandas as pd

ld = load_dataset("/Users/kmehant/Downloads/truth/", data_files="/Users/kmehant/Downloads/truth/validation-00000-of-00001.parquet", split="train")

sformat = """Question:
{q}

choices:
(a) {c1}
(b) {c2}
(c) {c3}

Answer: {a}"""

fdata = []

for data in ld:
    ai = random.randint(0, 2)
    choices = ["", "", ""]
    fdatadict = {}
    choices[ai] = data["mc1_targets"]["choices"][0]
    k = 1
    l = 0
    if ai == 0:
        aalpha = "(a)"
    if ai == 1:
        aalpha = "(b)"
    if ai == 2:
        aalpha = "(c)"
    for c in choices:
        if c == "":
            choices[l] = data["mc1_targets"]["choices"][k]
            k+= 1
        l+=1
    fdatadict["train_contents"] = sformat.format(q=data["question"], c1=choices[0], c2=choices[1], c3=choices[2], a=aalpha)
    fdatadict["q"] = sformat.format(q=data["question"], c1=choices[0], c2=choices[1], c3=choices[2], a="")
    fdatadict["a"] = aalpha
    fdata.append(fdatadict)

hf_dataset = Dataset.from_pandas(pd.DataFrame(data=fdata))

hf_dataset = hf_dataset.train_test_split(test_size=0.3)
hf_train = hf_dataset["train"]
hf_test = hf_dataset["test"]
hf_train.to_parquet("./train.parquet")
hf_test.to_parquet("./test.parquet") t