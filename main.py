from pathlib import Path

from sources.comparison import compare

data_root = Path("./original_data")
dataset = [
    {
        "path": data_root/"hiroshiba_normal_049.wav",
        "name": "hiho"
    },{
        "path": data_root/"tsuchiya_normal_049.wav",
        "name": "tsuchiya"
    }
]
# data_root = Path("./my_data")
# dataset = [
#     {
#         "path": data_root/"f2m.wav",
#         "name": "f2m"
#     },{
#         "path": data_root/"m2f.wav",
#         "name": "m2f"
#     }
# ]
for datum in dataset:
    compare(datum)
