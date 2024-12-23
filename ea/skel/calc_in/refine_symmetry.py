from ase import Atoms

def mhm_find_symmetry(
        atoms: Atoms,
        tolmin: float=1e-6,
        tolmax: float=1e-1,
        ntol: int=10,
        verbose: bool=False,
        refine: bool=False):
    """ Basically a copy of the MHM routine find_symmetry.
    Within a range of tolerances, try to find the highest
    space group for the material. It is not recommendable 
    to go much higher than 0.1 (even that might be pushing
    it).

    atoms: Atoms
        The structure to refine. ASE Atoms object.
    tolmin: float
        Minimum symprec to consider
    tolmax: float
        Maximum symprec to consider
    ntol: int
        Number of tolerances to consider
    verbose: bool
        Print information at every step?
    refine: bool
        Refine structure?

    Returns
        atoms | None : refined structure at the found tolerange
        int : space group found
        tol : tolerance used for space group found
    """
    from ase.spacegroup.symmetrize import refine_symmetry, check_symmetry
    from warnings import warn

    # Placeholder for spacegroup and tolerance
    spgcur = 0
    tolcur = 0

    # Multiplication factor to grow tolerance
    tolfact = (tolmax/tolmin)**(1./ntol)

    # Copy the input atoms object to avoid overwritting it when refining
    tmp_atoms = atoms.copy()

    for itol in range(1, ntol+1):
        #
        tol = tolmin*tolfact**itol
        spglib_set = check_symmetry(
            atoms= tmp_atoms,
            symprec= tol,
            verbose= False)
        #
        if (spglib_set is None):
            # Sometimes the symmetrization will crash, with spglib returning
            # a None object. This likely happens because the symprec is too
            # large and thus makes no sense to further increase it.
            # Print a warning and keep running at the previous tolerance
            message = f"spglib crashed for tolerance {tol:15.8f}; continuing with earlier tol"
            warn(message)
            break

        if (verbose):
            print ( f">>> Tolerance: {tol:15.8f}, space group: {spglib_set['number']:4d}")
        
        if (spglib_set["number"] > spgcur):
            spgcur = spglib_set["number"]
            tolcur = tol

    if refine:
        spglib_set = refine_symmetry(
            atoms= tmp_atoms,
            symprec= tolcur,
            verbose= True)
        return tmp_atoms, spglib_set["number"], tolcur

    return None, spgcur, tolcur



def refine_symmetry_w_energy(
        atoms: Atoms,
        calculator,
        tolmin: float=1e-6,
        tolmax: float=1e-1,
        ntol: int=10,
        verbose: bool=False):
    """ A variation of the symmetry refinning of Minhocao.
    Uses an array of tolerances and checks the space group
    computed by spglib. In addition also compares energies
    at fixed geometry which requires an adequate calculator.

    atoms: Atoms
        The structure to refine. ASE Atoms object.
    calculator: Calculator
        An ASE calculator object.
    tolmin: float
        Minimum symprec to consider
    tolmax: float
        Maximum symprec to consider
    ntol: int
        Number of tolerances to consider
    verbose: bool
        Print information at every step?
    refine: bool
        Refine structure?

    Returns
        atoms | None : refined structure at the found tolerange
        int : space group found
        tol : tolerance used for space group found
    """
    from ase.spacegroup.symmetrize import refine_symmetry, check_symmetry
    from warnings import warn

    # Placeholder for spacegroup and tolerance
    items = {
        "energy": [],
        "tolerance": [],
        "spg_num": [] }

    # Multiplication factor to grow tolerance
    tolfact = (tolmax/tolmin)**(1./ntol)

    for itol in range(1, ntol+1):
        #
        tol = tolmin*tolfact**itol
        
        # Copy the input atoms object to avoid overwritting it when refining
        tmp_atoms = atoms.copy()

        # Refine the structure 
        spglib_set = refine_symmetry(
            atoms= tmp_atoms,
            symprec= tol,
            verbose= verbose)
        #
        if (spglib_set is None):
            # Sometimes the symmetrization will crash, with spglib returning
            # a None object. This likely happens because the symprec is too
            # large and thus makes no sense to further increase it.
            # Print a warning and keep running at the previous tolerance
            message = f"spglib crashed for tolerance {tol:15.8f}; exiting loop"
            warn(message)
            break

        # Now that we have the refined structure, compute the energy 
        # at this geometry.
        tmp_atoms.calc = calculator
        calculator.calculate(tmp_atoms)

        # TODO: mace calculator is returning a list intead of a float...
        try:
            energy = tmp_atoms.get_potential_energy()[0]/len(tmp_atoms)
        except:
            energy = tmp_atoms.get_potential_energy()/len(tmp_atoms)

        # Append items
        items["energy"].append(energy)
        items["tolerance"].append(tol)
        items["spg_num"].append(spglib_set["number"])

    return items
