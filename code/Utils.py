from collections import defaultdict
from typing import Tuple, List, Dict
import urllib.request

from Bio import pairwise2
from Bio.PDB import PDBParser, NeighborSearch, PDBIO, Select, Residue
from Bio.PDB.Selection import unfold_entities
from openmm.app import PDBFile
from pdbfixer import PDBFixer

from SASA import ShrakeRupley

PROT_3TO1 = {
    "CYS": "C",
    "ASP": "D",
    "SER": "S",
    "GLN": "Q",
    "LYS": "K",
    "ILE": "I",
    "PRO": "P",
    "THR": "T",
    "PHE": "F",
    "ASN": "N",
    "GLY": "G",
    "HIS": "H",
    "LEU": "L",
    "ARG": "R",
    "TRP": "W",
    "ALA": "A",
    "VAL": "V",
    "GLU": "E",
    "TYR": "Y",
    "MET": "M",
}


def download_pdb(pdb_id: str, save_path: str) -> None:
    """Downloads a PDB file given its accession code

    Args:
        pdb_id (str): 4-letter PDB ID
        save_path (str): Path to target directory.
    """
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    urllib.request.urlretrieve(url, save_path)

    return None


def fix_pdb_structure(
    input_path: str,
    output_path: str,
    exclude_termini: bool = True,
    keepWater: bool = False,
) -> None:
    """Sanitises a PDB structure by building missing density and removing
       heterogens/waters. Can be configured to not build missing residues at termini,
       since these often tend to be longer unresolved tails.

    Args:
        input_path (str): Path to input PDB file
        exclude_termini (bool, optional): Whether to not build terminal missing density.
                                          Defaults to True.
        keepWater (bool, optional): Whether to keep water molecules. Defaults to False.

    Returns:
        None
    """

    # Initialise fixer and find missing residues based on SEQRES records
    fixer = PDBFixer(filename=input_path)
    fixer.findMissingResidues()

    # Exclude termini by identifying missing stretches in a chain that start with resid
    # 0 (N-terminal density) or resid > the length of present residues in the
    # chain (C-terminal density)
    if exclude_termini:
        exclude_from_build = []

        for missing_residues_id in fixer.missingResidues.keys():
            chain_id, missing_residue_start_id = missing_residues_id
            n_residues_chain = len(
                list(list(fixer.topology.chains())[chain_id].residues())
            )
            if (
                missing_residue_start_id == 0
                or missing_residue_start_id >= n_residues_chain
            ):
                exclude_from_build.append(missing_residues_id)

        for excluded_id in exclude_from_build:
            del fixer.missingResidues[excluded_id]

    # Proceed to add missing atoms for the chosen residues.
    fixer.findMissingAtoms()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.addMissingAtoms()
    fixer.removeHeterogens(keepWater=keepWater)

    # Save the fixed PDB file
    with open(output_path, "w") as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f)

    return None


def count_start_gaps(aligned_seq: str) -> int:
    """Counts the number of gaps ("-" characters) at the N-term of an aligned sequence.

    Args:
        aligned_seq (str): Aligned sequence, where gaps are "-" characters

    Returns:
        int: number of gaps at the N-terminus
    """
    count = 0
    for char in aligned_seq:
        if char == "-":
            count += 1
        else:
            break
    return count


def make_equivalent_chains(
    complex1_filepath: str,
    complex2_filepath: str,
    complex1_chain: str,
    complex2_chain: str,
) -> Tuple[int, str]:
    """For a polypeptide of interest in two structures, this function identifies the
       sequence of each and aligns the two, creating a union of the two sequences.
       For example, "AGHT" and "GHTYS" would result in "AGHTYS".

       The resultant sequence is then numbered from 1, and the numbering is transferred
       on to the chain of interest in both of the structures such that equivalent
       residues have the same numbering in both structures.

       NB: The sequences must be identical except for missing residues.
       That is, there may be no substitutions.

    Args:
        complex1_filepath (str): Path to PDB file 1
        complex2_filepath (str): Path to PDB file 2
        complex1_chain (str): Chain ID of chain of interest in file 1
        complex2_chain (str): Chain ID of chain of interest in file 2

    Returns:
        Tuple: a tuple containing the number of residues in the merged sequence and
               the merged sequence itself.
    """

    # Load the two complexes
    parser = PDBParser(QUIET=True, PERMISSIVE=True)

    s1 = parser.get_structure("s1", complex1_filepath)
    s2 = parser.get_structure("s2", complex2_filepath)

    # Extract sequences based on the atom records of the structures
    seq1, seq2 = "", ""
    for residue in s1[0][complex1_chain]:
        seq1 += PROT_3TO1[residue.resname]

    for residue in s2[0][complex2_chain]:
        seq2 += PROT_3TO1[residue.resname]

    # Align the two sequences and see if either sequence has a more residues in the N-term
    alignments = pairwise2.align.globalxx(seq1, seq2)
    seq1_aligned, seq2_aligned = alignments[0].seqA, alignments[0].seqB
    start_gaps_seq1 = count_start_gaps(seq1_aligned)
    start_gaps_seq2 = count_start_gaps(seq2_aligned)

    # Compile the merged sequence
    total_n_residues = len(seq1_aligned)
    merged_seq = ""
    for res_seq1, res_seq2 in zip(seq1_aligned, seq2_aligned):
        if res_seq1 == "-":
            merged_seq += res_seq2
        else:
            merged_seq += res_seq1

    # Renumber residues in the structures such that they become equivalent
    if start_gaps_seq1 > 0:
        for res in s1[0][complex1_chain].get_residues():
            curr_resnum = res.id[1]
            res.id = (" ", curr_resnum + start_gaps_seq1, " ")

    if start_gaps_seq2 > 0:
        for res in s2[0][complex1_chain].get_residues():
            curr_resnum = res.id[1]
            res.id = (" ", curr_resnum + start_gaps_seq1, " ")

    # Save the two structures with adjusted numbering on the chains of interest
    io = PDBIO()
    io.set_structure(s1)
    io.save(complex1_filepath)
    io.set_structure(s2)
    io.save(complex2_filepath)

    return total_n_residues, merged_seq


def get_interface_residues(
    filepath: str, prot_A_chain: str, prot_X_chains: List[str], distance: float = 4.5
) -> Dict[int, bool]:
    """
    For some chain A in a PDB file in complex with other chains of interest, returns the
    identities of residues in chain A that have heavy atoms within some distance of any
    heavy atoms of the other chains of interest.

    Args:
        filepath (str): Path to PDB file
        prot_A_chain (str): ID of chain on which to find interacting residues.
        prot_X_chains (str): A list of chain IDs based on whose residues residues of
                             chain A will be evaluated as interface/non-interface.
        distance (float, optional): The heavy atom distance cutoff (in Angstrom)
                                    within which to residues are considered to be
                                    interacting with one another. Defaults to 4.5.

    Returns:
        Dict[int, bool]: A dictionary of whether a residue in chain A is part of the
                         interface. Keys are the residue numbers of residues.
    """

    # Load the structure and initialise empty list to store interface residues
    parser = PDBParser(QUIET=True, PERMISSIVE=True)
    s = parser.get_structure("s", filepath)[0]
    interface_residues = []

    # Get heavy atoms of both prot_A and prot_X chains
    prot_A_atoms = [
        a
        for a in s.get_atoms()
        if a.get_full_id()[2] == prot_A_chain and a.element != "H"
    ]
    prot_X_models = [s[chain] for chain in prot_X_chains]
    prot_X_atoms = unfold_entities(prot_X_models, "A")
    prot_X_atoms = [a for a in prot_X_atoms if a.element != "H"]

    # For each heavy atom of prot_X
    for X_atom in prot_X_atoms:
        # Find neighbours within atoms of A
        ns = NeighborSearch(prot_A_atoms)
        close_residues = ns.search(X_atom.coord, distance, level="R")
        # Add the residues of close atoms of A to the interface
        for r in close_residues:
            if r not in interface_residues:
                interface_residues.append(r)

    # Create and populate the dictionary to label residues as interface or not.
    # Use a defaultdict in case a residue is only present in one of the PDB structures.
    is_interface = defaultdict(lambda: False)

    for res in s[prot_A_chain].get_residues():
        if res in interface_residues:
            is_interface[res.id[1]] = True
        else:
            is_interface[res.id[1]] = False

    return is_interface


class ChainSelect(Select):
    def __init__(self, chain_ids: List[str]):
        self.chain_ids = chain_ids

    def accept_chain(self, chain: str):
        return chain.id in self.chain_ids


def save_chain_of_interest(
    input_path: str, chain_ids: List[str], output_path: str
) -> None:
    """Dice a structure, saving only chains of interest to a new PDB file

    Args:
        input_path (str): Path to input structure
        chain_ids (List[str]): List of chain IDs we want to retain
        output_path (str): Path to output file

    Returns:
        None
    """

    # Load the structure, directly accessing the first model
    parser = PDBParser(QUIET=True, PERMISSIVE=True)
    structure = parser.get_structure("s", input_path)[0]

    # Save just the chains of interest
    io = PDBIO()
    io.set_structure(structure)
    select = ChainSelect(chain_ids)
    io.save(output_path, select=select)


def calculate_residue_bsa(
    res_monomer: Residue, res_complex: Residue
) -> Tuple[float, float]:
    """For a residue, calculates the buried surface area (BSA) decomposed into side
       chain (sc) and backbone (bb) by calculating the SASA in the residue in the bound
       structure, and subtracting the SASA of the residue in the monomeric structure.

    Args:
        res_monomer (Residue): Biopython residue from the monomeric structure, must have
                               the .sasa attribute already present.
        res_complex (Residue): Biopython residue from the bound structure, must have
                               the .sasa attribute already present.

    Returns:
        Tuple[float, float]: BSAs of backbone and side chain for the residue
    """

    # Initialise bb and sc values for monomeric and complexes SASAs for the residue
    res_monomer_bb, res_complex_bb = 0.0, 0.0
    res_monomer_sc, res_complex_sc = 0.0, 0.0

    # For each atom in the monomeric residue
    for atom in res_monomer.get_atoms():
        # Add the bb and sc SASA to the total values
        if atom.name in ["N", "CA", "H", "C", "O"]:
            res_monomer_bb += atom.sasa
        else:
            res_monomer_sc += atom.sasa

    # For each atom in the complexed residue
    for atom in res_complex.get_atoms():
        # Add the bb and sc SASA to the total values
        if atom.name in ["N", "CA", "H", "C", "O"]:
            res_complex_bb += atom.sasa
        else:
            res_complex_sc += atom.sasa

    # Calculate the BSA, BSA = SASA_monomer - SASA_complex
    bb_bsa = res_monomer_bb - res_complex_bb
    sc_bsa = res_monomer_sc - res_complex_sc

    return bb_bsa, sc_bsa


def get_by_res_BSA(
    monomer_filepath: str, complex_filepath: str, chain_id: str
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """For a given chain, calculates the buried surface area of each residue,
       decomposing it into side chain (sc) and backbone (bb).

    Args:
        monomer_filepath (str): Path to input file with only the chain of interest present.
        complex_filepath (str): Path to input file with the complexed structure present.
        chain_id (str): ID of chain of interest

    Returns:
        Tuple[np.ndarray, np.ndarray]: dictionaries containing each of the residues of
                                       the chain of interest, populated with
                                       the BSA of each residue. The first tuple refers
                                       to backbone values, the second to side chains.
    """

    # Load the structures
    parser = PDBParser(QUIET=True, PERMISSIVE=True)
    s_monomer = parser.get_structure("monomer", monomer_filepath)[0]
    s_complex = parser.get_structure("complex", complex_filepath)[0]

    # Calculate BSAs at the atom level
    sr = ShrakeRupley()
    sr.compute(s_monomer, level="A")
    sr.compute(s_complex, level="A")

    # Create dictionaries to store BSA values for each residue
    # Use a defaultdict in case a residue is only present in one of the PDB structures.
    bb_bsas, sc_bsas = defaultdict(lambda: 0.0), defaultdict(lambda: 0.0)

    # For each residue, calculate the total BSA and populate the dictionaries
    for res_monomer, res_complex in zip(
        s_monomer[chain_id].get_residues(), s_complex[chain_id].get_residues()
    ):
        bb_bsa, sc_bsa = calculate_residue_bsa(res_monomer, res_complex)

        bb_bsas[res_monomer.id[1]] = bb_bsa
        sc_bsas[res_monomer.id[1]] = sc_bsa

    return bb_bsas, sc_bsas


def replace_bfactors(
    input_path: str, output_path: str, values: Dict[int, float]
) -> None:
    """Populates the B-factor column of a PDB file with some other quantity

    Args:
        input_path (str): Path to input PDB file.
        output_path (str): Path to output PDB file.
        values (dict): Dict where keys correspond to residue numbers of the chain of
                        interest and values are the new B-factors.

    Returns:
        None
    """

    # Load structure
    parser = PDBParser(QUIET=True, PERMISSIVE=True)
    struct = parser.get_structure("struct", input_path)[0]
    # Repopulate the B-factors of the Biopython structure
    for res in struct.get_residues():
        res_id = res.get_id()[1]
        for atom in res.get_atoms():
            atom.set_bfactor(values[res_id])
    # Save the current structure
    io = PDBIO()
    io.set_structure(struct)
    io.save(output_path)
