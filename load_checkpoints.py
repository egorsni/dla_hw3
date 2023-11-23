import gdown
import shutil
import os
from pathlib import Path


URL_LINKS = {
    'model_best': 'https://drive.google.com/uc?id=1qZEDg83wrltCvDu3iUXC3qFJbeoXgySv'
}

def main():
    dir = Path(__file__).absolute().resolve().parent
    for name in URL_LINKS:
        
        checkpoint_dir = dir / 'saved' / 'models' / 'one_batch_test'/ 'checkpoints' / 'model_best'
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
        zip_pth = checkpoint_dir / ('model_best.zip')
        model_pth = checkpoint_dir / 'model_weights.pth'
        if not model_pth.exists():
            gdown.download(URL_LINKS[name], str(zip_pth), quiet=False)
            shutil.unpack_archive(str(zip_pth), str(checkpoint_dir), "zip")
            os.remove(zip_pth)

if __name__ == "__main__":
    main()