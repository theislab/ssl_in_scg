import os
import torch


def load_last_checkpoint(CHECKPOINT_PATH):
    """
    In CHECKPOINT_PATH, load the 'last_checkpoint.ckpt' file of the highest version
    The files are named '.../default/version_x/last_checkpoint.ckpt'
    Get the highest x and return the path to the checkpoint
    :param CHECKPOINT_PATH: Path to the checkpoint
    :return: Path to the latest checkpoint
    """
    ckpt_dirs = []
    versions = []
    for root, dirs, files in os.walk(CHECKPOINT_PATH):
        for file in files:
            if file.endswith('.ckpt') and 'last' in file:
                # get the path to the checkpoint
                checkpoint_path = os.path.join(root, file)
                print('Found checkpoint: ', checkpoint_path)
                ckpt_dirs.append(checkpoint_path)
                # versions.append(int(checkpoint_path.split('version_')[1].split('/')[0]))

    # get the index of the highest version in the versions list
    # highest_version_index = versions.index(max(versions))
    # latest_ckpt = ckpt_dirs[highest_version_index]
    latest_ckpt = ckpt_dirs[0]
    return latest_ckpt


def load_best_checkpoint(CHECKPOINT_PATH, val=False):
    """
    Load the best checkpoint from the given path
    :param CHECKPOINT_PATH: Path to the checkpoint
    :param val: If True, load the best checkpoint based on validation loss, else load best checkpoint in general
    :return: Path to the latest checkpoint
    """
    # loop over all '.ckpt' files in the subdirectories
    ckpt_dirs = []
    for root, dirs, files in os.walk(CHECKPOINT_PATH):
        for file in files:
            if val:
                if file.endswith('val.ckpt'):
                    # get the path to the checkpoint
                    checkpoint_path = os.path.join(root, file)
                    print('Found checkpoint: ', checkpoint_path)
                    ckpt_dirs.append(checkpoint_path)
            else:
                if file.endswith('.ckpt'):
                    # get the path to the checkpoint
                    checkpoint_path = os.path.join(root, file)
                    print('Found checkpoint: ', checkpoint_path)
                    ckpt_dirs.append(checkpoint_path)

    if len(ckpt_dirs) == 0:
        raise ValueError('No checkpoints found in directory {}'.format(CHECKPOINT_PATH))
    elif len(ckpt_dirs) == 1:
        print('Only one checkpoint found in directory {}'.format(CHECKPOINT_PATH))
        return ckpt_dirs[0]

    # get the best checkpoint
    for i in range(len(ckpt_dirs)):
        if i == 0:
            ckpt = torch.load(ckpt_dirs[i])
            val_key = list(ckpt['callbacks'].keys())[1]
            best_val_loss = abs(ckpt['callbacks'][val_key]['best_model_score'].data.item())
            best_ckpt = ckpt_dirs[i]
        else:
            ckpt = torch.load(ckpt_dirs[i])
            val_key = list(ckpt['callbacks'].keys())[1]
            val_loss = abs(ckpt['callbacks'][val_key]['best_model_score'].data.item())
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_ckpt = ckpt_dirs[i]

    print('Will load checkpoint from {} with val loss {:.4f}'.format(best_ckpt, best_val_loss))
    return best_ckpt


def checkpoint_exists(CHECKPOINT_PATH):
    """
    Check if a checkpoint exists in the given path or not
    :param CHECKPOINT_PATH: Path to the checkpoint
    :return: True if checkpoint exists, False if not
    """
    valid_checkpoint = False
    for root, dirs, files in os.walk(CHECKPOINT_PATH):
        for file in files:
            if file == 'best_checkpoint_val.ckpt':
                valid_checkpoint = True
                print('Found valid checkpoint at directory', CHECKPOINT_PATH)
                return valid_checkpoint
    return valid_checkpoint


def check_target_size(train_dataloader):
    """
    Check the number of unique classes in the PyTorch dataloader
    Loop over 10 epochs to get a good estimate of the number of classes
    :param train_dataloader: PyTorch dataloader
    :return: Number of classes
    """
    # get number of classes
    num_classes = 0
    for i in range(10):
        for batch in train_dataloader:
            num_classes = max(num_classes, batch['target'].max().item())
    return num_classes + 1


def find_subdirectories(root: str) -> list:
    """
    Find all subdirectories in root with the structure of root + a + b + c + 'default' + 'version_x' + 'checkpoints' + 'best_checkpoint_val.ckpt'
    The list contains the different versions of the subdirectory a + b + c
    Each combination should only appear once and only as a full root + a + b + c but without the 'default' + 'version_x' + 'checkpoints' + 'best_checkpoint_val.ckpt'
    :param root: Path to the root directory
    :return: List  of subdirectories
    """
    subdirectories = []
    for root, dirs, files in os.walk(root):
        for cur_dir in dirs:
            if cur_dir.startswith('version_'):
                subdirectories.append('/' + root.strip('/default'))

    for i in range(50):
        gridsearch_model = '_v' + str(i)
        gridsearch_model_a = '_v' + str(i) + 'a'
        subdirectories = [x for x in subdirectories if gridsearch_model not in x]  # remove gridsearch models
        subdirectories = [x for x in subdirectories if gridsearch_model_a not in x]  # remove gridsearch models
        subdirectories = [x for x in subdirectories if 'CN_' in x]  # remove non-CellNet models
        subdirectories = list(dict.fromkeys(subdirectories))

    subdirectories.sort()
    print('Found {} subdirectories'.format(len(subdirectories)) + '\n')
    return subdirectories


def list_checkpoint_in_subdirectory(subdirectory: str) -> list:
    """
    Given a subdirectory, return a list of all directories of files that end with '.ckpt'
    :param subdirectory: String of subdirectory
    :return: List of all directories of files that end with '.ckpt'
    """
    checkpoint_list = []
    for root, dirs, files in os.walk(subdirectory):
        for file in files:
            if file.endswith('.ckpt'):
                checkpoint_list.append(os.path.join(root, file))
    return checkpoint_list
