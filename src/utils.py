import os
import dill

def save_model(path, model):
    
    dir_path = os.path.dirname(path)

    os.makedirs(dir_path, exist_ok = True)

    with open(path, "wb") as file_obj:
        dill.dump(model, file_obj)
    print("Model saved !")