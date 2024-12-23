
def is_connected(
        structure,
        epsilon: float= 1e-6,
        radii= None,
        print_fiedler: bool= False,
        mask: list=[True, True, True] ):
    """
    Routine to determine if a given structure is connected. This depends on what we mean by connected.
    For example, a molecule is connected if we consider it an isolated structure.
    However, from the point of view of the periodic supercell, each molecule is isolated and thus
    the structure is *not* connected.

    These two aspects can be consolidated by whether making a supercell or not.
    In the previous example, making a supercell will give a disconnected result.
    
    We add a mask so that the supercell can be built in isolated directions, useful for 2D systems.

    Spheres are centered in each atom with radius equal to their covalent radii.
    If the sphere overlap the atoms are considered connected.
    From this get the adjacency matrix (using ASE) and the degree matrix (manually).
    Build the Laplacian matrix.

    L = D - A

    Get eigenvalues and check the Fiedlers value.
    If the Fiedler value if 0 (within numercial accuracy) then the graph is disconnected

    Arguments:
        struc (in):
            ASE object
        epsilon (in) ::
            float, the threshold for comparison of the Fiedler value
        radii (in)  ::
            dictionary of covalent radii for the analysis. If None, then
            ASE's values are used.
        print_fiedler (in) ::
            bool, print more info?
    """
    from ase import neighborlist
    from ase.build.supercells import make_supercell
    from ase.neighborlist import natural_cutoffs
    import numpy as np
    from numpy.linalg import eig

    # Build the supercell
    sup_matrix = [ [ max(2*mask[0] , 1), 0, 0], [ 0, max(2*mask[1],1 ), 0 ], [0 ,0, max(2*mask[2],1) ] ]
    supercell  = make_supercell( structure, sup_matrix )

    # Check if the covalent radii are given
    if ( radii is None ):
        # Cutoffs are by default the covalent radius
        cutOff = natural_cutoffs( supercell, mult = 1 )
    else:
        # Verify if given dictionary of radii contains values for all elements in struc
        assert( np.any( list( i in struc.get_chemical_symbols() for i in radii.keys() ) ) )
        # Set the cutoffs
        cutOff = [ radii[isymbol] for isymbol in list(struc.symbols) ]


    # Get the neighbour list
    # Get connectivity matrix. (i,j) = 1 is connected, (i,j) = 0 is unconnected
    nl = neighborlist.NeighborList( cutOff, bothways=True, self_interaction=False )
    nl.update( supercell )
    adjency_matrix = nl.get_connectivity_matrix( sparse=False )

    # Build manually the degree matrix
    # TODO: This should be updated...
    degree_matrix = np.zeros( (len(supercell), len(supercell) ) )

    for i in range( len( supercell ) ):
        # For some reason self_interaction=False still connects to periodic copies of the atoms...
        # Remove this manually
        adjency_matrix[i,i] = 0.0

        # Get degree matrix (jusst count the umber of connections per atom from the adjency matrix
        degree_matrix[i,i] = sum( adjency_matrix[i, j] for j in range( len(supercell) ) if (j!=i) )
    
    # Print relevant information
    if ( print_fiedler ):
        print (" Adjency Matrix ") 
        print ( adjency_matrix )
        print ("\n") 
        #
        print (" Degree Matrix ") 
        print ( degree_matrix )
        print ("\n") 

    # build Laplacian matrix
    # From definition L = D - A   
    laplacian_matrix =  ( degree_matrix - adjency_matrix )
    #
    if ( print_fiedler ):
        print (" Laplacian Matrix ") 
        print ( laplacian_matrix )
        print ("\n") 

    # Get Fiedler value
    w, v = eig(laplacian_matrix)

    # L is definite positive.
    # Second eigenval is zero (within numericall accuracy) if graph is disconnected
    if ( print_fiedler ):
        print (" Fiedler's Value {0:6.3f}".format( np.sort(w)[1] ) )
        #
        return np.sort(w)[1] > epsilon
    #
    return np.sort(w)[1] > epsilon



def get_min_dist(
        atoms,
        cutoff: float=10.):
    """ Compute the smallest distance between pairs
    of chemical species for a given structure
    
    Arguments:
        atoms: Ase.atoms
            Target structure to analyse as an ASE atoms
            object
        cutoff: float
            The cutoff for the generation of neighbours
            
    Returns:
        min_dis: dict
            A dictionary whose keys are pairs of elements
            (as strings of the symbol)
    """
    from ase.neighborlist import neighbor_list
    from itertools import combinations
    from collections import Counter
    import numpy as np

    # Returns a list i with the index of 1st atom,
    # j with index of 2nd atom, and d the interatomic
    # distance between them
    i, j, d = neighbor_list('ijd', atoms, cutoff)
    
    # Iterate over elements, find corresponding indices in the structure
    indices = {x: [atom.index for atom in atoms if atom.symbol==x] for x in atoms.symbols}
    
    # Iterate over pairs of elements
    pairs = Counter(combinations(atoms.symbols, r=2)).keys()
    min_dist = {}
    
    for pair in pairs:
        # Locate which entries of i are for element pair[0]
        where_i = np.logical_or.reduce([i==a for a in indices[pair[0]]])
        where_j = np.logical_or.reduce([j==a for a in indices[pair[1]]])
        
        # The interception then gives the indices for d
        where_d = np.logical_and( where_i, where_j)
        min_dist[pair] = np.min( d[where_d] )
        
    return min_dist




if __name__=="__main__":
    #
    import sys
    import ase.io
    #
    # TODO: add proper parser inputs
    filename = sys.argv[1]
    #
    structure = ase.io.read( filename )
    # 
    print ( "As 3D     : ", is_connected( structure, print_fiedler = False ) )
    print ( "As 2D     : ", is_connected( structure, print_fiedler = False , mask=[True, True, False] ) )
    print ( "As 0D     : ", is_connected( structure, print_fiedler = False , mask=[False, False, False] ) )
