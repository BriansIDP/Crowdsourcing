import json
import sys


# with open("crosscheck_NLI_mistral_zephyr.json") as fin:
#     data1 = json.load(fin)
# with open("crosscheck_NLI_mistral_llama2.json") as fin:
#     data2 = json.load(fin)
# with open("crosscheck_NLI_mistral_llama.json") as fin:
#     data3 = json.load(fin)
# with open("crosscheck_NLI_mistral_mistral.json") as fin:
#     data4 = json.load(fin)
# with open("crosscheck_NLI_mistral_beluga.json") as fin:
#     data5 = json.load(fin)
# with open("crosscheck_NLI_mistral_vicuna.json") as fin:
#     data6 = json.load(fin)
# with open("crosscheck_NLI_mistral_gpt3.json") as fin:
#     data7 = json.load(fin)
infile = sys.argv[1]
sidefile = sys.argv[2]

with open(infile) as fin:
    datain = json.load(fin)

with open(sidefile) as fin:
    dataside = json.load(fin)

data = []
for i, datapiece in enumerate(datain):
    fullpiece = {**datapiece, **dataside[i]}
    # fullpiece = {**datapiece, **data2[i], **data3[i], **data4[i], **data5[i], **data6[i], **data7[i]}
    data.append(fullpiece)

with open(infile, "w") as fout:
    json.dump(data, fout, indent=4)
