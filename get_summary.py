def check_too_close( min_dist, radii=None, cutoff=10):
    """ Check if the interatomic distances are too close.
    Usefull for the NN. The criterium is rather arbitrary
    though, since I do not know the correlation between
    NN and DFT bond lenghts...
    """
    import numpy as np
    from is_connected import get_min_dist
    from ase.data import covalent_radii, atomic_numbers
    #
    if radii is None:
        radii = covalent_radii
    #
    for pair in min_dist.keys():
        radius_i = radii[ atomic_numbers[pair[0]] ]
        radius_j = radii[ atomic_numbers[pair[1]] ]
        # The comparison distance
        comp_dist = np.max( [radius_i, radius_j])*1.1
        #
        if (min_dist[pair] < comp_dist):
            return True

    return False


def select_unique_structures( list_of_alloys, energy_thresh = 9999999999999 ):
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from tqdm import tqdm

    # Define the structure matcher
    stm = StructureMatcher()

    # A placeholder for the unique structures
    list_unique_alloys = []

    # Now iterate over the list of all alloys
    for i in tqdm(list_of_alloys, total= len(list_of_alloys)):
        unique = True

        # Iterate over the unique structures so far
        for k in list_unique_alloys[::-1]:
            # Energy difference must be within threshold
            if (abs(i["eatom"] - k["eatom"]) < energy_thresh):
                if ( stm.fit( struct1 = i["structure"], struct2 = k["structure"], symmetric = False ) ):
                    unique = False
                    break
        if unique:
            list_unique_alloys.append(i)

    return list_unique_alloys



if __name__=="__main__":
    import pickle
    import glob
    import sys

    import pandas as pd
    from remote_calculator import RemoteCalculator
    
    from pymatgen.io.ase import AseAtomsAdaptor
    from tqdm import tqdm

    runs_folder = sys.argv[1]

    # iterate over runs and load individual pickles
    tmp = []
    count = 0
    
    folders = glob.glob( runs_folder+"/[0-9]*" )
    for i in tqdm( folders, total=len(folders)):
        try:
            with open( f"{i}/data.pkl.bz2", "rb") as openfile:
                data = pickle.load(openfile)
            data["file"] = i
            tmp.append(data)
    
        except Exception as e:
            count+=1

    print (f"There are {count/(count+len(tmp))*100:.2f}% of unexplained chrashes")

    # Build the dataframe
    df = pd.DataFrame( tmp )
    print (f"{df.shape[0]} entries")
    df["too_short"] = df["min_dis"].apply(lambda x: check_too_close(min_dist= x))
    print (f"{df[ df['connected'] == False ].shape[0]} disconnected structures")
    print (f"{df[ df['converged_ionic'] == False ].shape[0]} unconverged structures")
    print (f"{df[ df['too_short'] == True ].shape[0]} structures with small interatomic distances")

    #
    df_safe= df[ (df['connected'] == True) & (df['converged_ionic'] == True) & (df['too_short'] == False) ].sort_values( by="eatom")
    df_safe.reset_index(inplace=True, drop=True)
    df_safe["structure"] = df_safe["atoms"].apply( lambda x: AseAtomsAdaptor.get_structure(atoms= x) )

    #
    uniques = select_unique_structures(df_safe.to_dict("records"), energy_thresh = 20e-3)
    df_uniques = pd.DataFrame( uniques )
    df_uniques.to_pickle(f"summary.pkl.bz2")

    print (f"Found {len(df_uniques)} unique structures")
