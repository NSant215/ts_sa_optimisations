import numpy as np
import eggholder as egg


def init_x(n,d):
    '''Chooses starting point x by randomly choosing d 
    possible starting points and picking the one that minimises
    f most.

    inputs
        d(int): dimension of x
        n(int): number of compared starting values
      
    returns the best x of the randomly chosen starting values
    '''
    xbest = np.zeros(d)
    fxbest = egg.eggholder(xbest)
    for i in range(n):
        x_try = np.random.rand(d)*1024-512
        fx_try = egg.eggholder(x_try)
        if fx_try < fxbest:
            xbest = x_try
            fxbest = fx_try
    return xbest

L = 25 # max number of archived solutions

def update_archive(x,fx,archive):
    '''Updates the archive if necessary.
    inputs
        x:          trial point
        fx:         cost function evaluated at x
        archive:    current set of archived solutions
    
    returns updated set of archived solutions and a boolean on whether
    the archive set has changed or not.
    '''
    min_val_found = False
    archive_updated = False

    listed_archive = [*archive]
    best_archived = min(listed_archive)
    worst_archived = max(listed_archive)
    l = len(listed_archive)
    assert l <= 25

    if fx < best_archived:
        min_val_found = True
    # dissimilarity parameters
    D_min = 60
    D_sim = 6

    # check if not dissimilar to any previously archived solution
    close_archived_point = None
    for fx0 in archive:
        D = np.linalg.norm(archive[fx0]-x)
        if D < D_min:
            close_archived_point = fx0
            if D < D_sim and fx < fx0: # meets archive criterion (4)
                del archive[fx0]
                archive[fx] = x
                archive_updated = True
                return archive, archive_updated

    if close_archived_point == None:
        if l == 25: # archive criterion (2) satisfied
            del archive[worst_archived] 
        archive[fx] = x # archive criterion (1) satisfied
        archive_updated = True
        return archive, archive_updated
    elif min_val_found: # archive criterion (3) satisfied
        del archive[close_archived_point]
        archive[fx] = x
        archive_updated = True
        return archive, archive_updated
    else:
        return archive, archive_updated