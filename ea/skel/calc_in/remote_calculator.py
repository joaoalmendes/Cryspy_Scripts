import numpy as np
import json
import requests
from ase import Atoms
from typing import Optional, Union
from ase.calculators.calculator import Calculator


def jsonize(atoms_dict):
    """ Cannot convert np.arrays directly into json, this fixes it
    """
    atoms_dict["numbers"] = atoms_dict["numbers"].tolist()
    atoms_dict["positions"] = atoms_dict["positions"].tolist()
    atoms_dict["cell"] = atoms_dict["cell"].tolist()
    atoms_dict["pbc"] = atoms_dict["pbc"].tolist()
    return  json.dumps(atoms_dict)


class RemoteCalculator(Calculator):
    """ This is a class that directly inherits the default Calculator and simply adds 
    the connection string as an atribute.

    Since that should be fixed on a run everything is static.
    
    It is preferable to treat *only* the one-shot calculation on the server side.
    This avoids problems with time-outs due to long optimizations and makes writting logs easier.
    The connection itself should not take long (hopefully but un-benchmarked).
    """
    implemented_properties = ["energy", "free_energy", "forces", "stress"]
    nolabel = True

    def __init__(self, connection, **kwargs):
        Calculator.__init__(self, **kwargs)
        self.connection = connection

    def calculate(
        self,
        atoms: Optional[Atoms] = None,
        properties: Optional[list] = None,
        system_changes: Optional[list] = None,
        thr_forces: float|None= None,
    ):
        """A copy of the native calculate function that instead of calling the
        executable performs a request.

        Args:
            atoms (ase.Atoms): ase Atoms object
            properties (list): list of properties to calculate
            system_changes (list): monitor which properties of atoms were
                changed for new calculation. If not, the previous calculation
                results will be loaded.
            thr_forces: float|None
                if a value is given, the calculator will check if the forces
                returned are in absolute value larger than thr_forces. If so
                raise a ValueError
        Returns:
            results (dict) 
        """
        # Create a copy of atoms input whithout any of the whistles and bells attached
        # as the constrains cannot be jasonfied and the calculator only needs the structure
        # and the calculator only returns the forces/stresses/energies.
        atoms_copy = atoms.copy()
        del atoms_copy.constraints

        #?????
        Calculator.calculate(self, atoms_copy, properties, system_changes)
        for name in self.implemented_properties:
            self.results.pop(name, None)

        # Do the request
        results = requests.post(self.connection, json=jsonize(atoms_copy.todict())).json()

        #
        if (thr_forces is not None):
            if (np.any( np.abs(np.array(results["forces"])) >= thr_forces)):
                raise ValueError("Calculator is returning very large forces. Could be a problem with the structure.")

        # Convert the stress/forces back to np.array
        results["forces"] = np.array(results["forces"])
        results["stress"] = np.array(results["stress"])
        self.results = results
