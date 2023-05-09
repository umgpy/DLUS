import os
from pathlib import Path
import time
import requests
import zipfile

def download_pretrained_weights(task_id):
    config_dir = Path(os.environ["RESULTS_FOLDER"]) / "nnUNet"
    (config_dir / "3d_fullres").mkdir(exist_ok=True, parents=True)
    config_dir = config_dir / "3d_fullres"
    
    if task_id == 108:
        weights_path = config_dir / "Task108_MABUSUrethra_DM_Danielsson"
        WEIGHTS_URL = "https://zenodo.org/record/7826416/files/Task108_MABUSUrethra_DM_Danielsson.zip?download=1"
    elif task_id == 112:
        weights_path = config_dir / "Task112_IGRTProstateVOI"
        WEIGHTS_URL = "https://zenodo.org/record/7826416/files/Task112_IGRTProstateVOI.zip?download=1"
    elif task_id == 113:
        weights_path = config_dir / "Task113_IGRTProstateVOI_Add"
        WEIGHTS_URL = "https://zenodo.org/record/7826416/files/Task113_IGRTProstateVOI_Add.zip?download=1"

    if WEIGHTS_URL is not None and not weights_path.exists():
        print(f"First-time user: Downloading pretrained weights for Task {task_id} (~2.3GB) ... It might take a while :-)")

        download_url_and_unpack(WEIGHTS_URL, config_dir)
        
def download_url_and_unpack(url, config_dir):
    import http.client
    # helps to solve incomplete read erros
    # https://stackoverflow.com/questions/37816596/restrict-request-to-only-ask-for-http-1-0-to-prevent-chunking-error
    http.client.HTTPConnection._http_vsn = 10
    http.client.HTTPConnection._http_vsn_str = 'HTTP/1.0'

    tempfile = config_dir / "tmp_download_file.zip"

    try:
        st = time.time()
        with open(tempfile, 'wb') as f:
            # session = requests.Session()  # making it slower
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                for chunk in r.iter_content(chunk_size=8192 * 16):
                    # If you have chunk encoded response uncomment if
                    # and set chunk_size parameter to None.
                    # if chunk:
                    f.write(chunk)

        print("Download finished. Extracting...")
        # call(['unzip', '-o', '-d', network_training_output_dir, tempfile])
        with zipfile.ZipFile(config_dir / "tmp_download_file.zip", 'r') as zip_f:
            zip_f.extractall(config_dir)
        print(f"  downloaded in {time.time()-st:.2f}s")
    except Exception as e:
        raise e
    finally:
        if tempfile.exists():
            os.remove(tempfile)
