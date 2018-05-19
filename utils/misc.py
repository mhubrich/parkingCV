import os


def list_files(path_dir, tag, absolute=True, whitelist=['.jpg', '.jpeg', '.png']):
    files = []
    for subdir in os.listdir(path_dir):
        if os.path.isdir(os.path.join(path_dir, subdir)):
            for f in os.listdir(os.path.join(path_dir, subdir)):
                if tag in f.lower():
                    if os.path.splitext(f)[-1] in whitelist:
                        if absolute:
                            files.append(os.path.join(path_dir, subdir, f))
                        else:
                            files.append(f)
    return files
