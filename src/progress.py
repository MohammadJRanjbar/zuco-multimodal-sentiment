"""Progress bars that render in Colab (`!python ...`) and terminals alike."""


def progress(iterable=None, **kwargs):
    from tqdm.auto import tqdm

    return tqdm(iterable, mininterval=1.0, dynamic_ncols=True, **kwargs)
