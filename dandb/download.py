import os
import tqdm
from zipfile import ZipFile
import requests

from . import MODELS_LINKS, MODELS_PATH

def in_jupyter_notebook():
    try:
        # Try to get the ipython shell name
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # In a Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # In a terminal running IPython
        else:
            return False  # Other types of environments
    except NameError:
        return False      # Not in IPython environment

def download_models():
    progress = tqdm.tqdm_notebook if in_jupyter_notebook() else tqdm.tqdm
    target_path = os.path.join(MODELS_PATH, "original")
    os.makedirs(target_path, exist_ok=True)
    downloaded_models = {}
    for model_name, model_link in progress(MODELS_LINKS.items()):
        model_path = os.path.abspath(os.path.join(target_path, model_name))
        if os.path.exists(model_path):
            downloaded_models["_".join(model_name.split('_')[:-1])] = model_path
            continue
        try: 
            target_path_zip =  os.path.join(target_path, "model_tmp.zip")
            # Send a GET request to the URL
            response = requests.get(model_link, stream=True)
            # Check if the request was successful
            if response.status_code == 200:
                # Open the file in binary write mode and save the contents
                with open(target_path_zip, 'wb') as file:
                    for chunk in response.iter_content(chunk_size=8192): 
                        file.write(chunk)
            else:
                raise Exception(f"Failed to download file. Status code: {response.status_code}")
            with ZipFile(target_path_zip, 'r') as zip_ref:
                zip_ref.extractall(target_path)
            downloaded_models["_".join(model_name.split('_')[:-1])] = model_path
        except Exception as e : 
            print('Could not download model %s'%model_link)
            print('Exception : \n\t%s'%e)
    return downloaded_models 
            