import numpy as np

def set_np_formatting():
    """ formats numpy print """
    np.set_printoptions(edgeitems=1, infstr='inf',
                        linewidth=4000, nanstr='nan', precision=2,
                        suppress=False, threshold=100, formatter=None)
