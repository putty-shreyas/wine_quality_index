import os
import dill

def save_obj(path, obj):
    
    dir_path = os.path.dirname(path)

    os.makedirs(dir_path, exist_ok = True)

    with open(path, "wb") as file_obj:
        dill.dump(obj, file_obj)
    print("Object saved !")

def load_obj(path):
    with open(path, "rb") as file_obj:
        return dill.load(file_obj)