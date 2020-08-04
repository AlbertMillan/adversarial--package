import torch

def centers(c_mm, n_classes, d):
    '''
    Computes centers for MMC loss.
    :param c_mm: constant
    :param d: dimension of feature space in PENULTIMATE layer
    :param n_classes: number of classes in dataset
    :return out: optimal mean vectors of size [n_classes]
    '''
    out = torch.zeros((n_classes, d))
    out[0,0] = 1

    for i in range(1, n_classes):

        for j in range(0, i):

            out[i, j] = -(1/(n_classes - 1) + torch.dot(out[i], out[j])) / out[j, j]

        out[i, i] = torch.sqrt( (1 - torch.norm(out[i])**2) )

    out *= c_mm
    return out

if __name__ == "__main__":
    print(centers(1,2,3))

