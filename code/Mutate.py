import pyrosetta
from pyrosetta import rosetta
from pyrosetta.rosetta.core.pose import Pose
import numpy as np
import os
import matplotlib.pyplot as plt
import multiprocessing

pyrosetta.init()

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


def relax_structure(input_path: str, output_path: str, n_repeats: int = 5) -> Pose:
    """Relaxes a Pose object using the FastRelax protocol. This involves rounds of
       side-chain packing and whole-atom-minimisations. For more detail on the protocol,
       refer to the Rosetta documentation. Saves the minimised structure as a PDB file.

    Args:
        pose (Pose): Input PDB file
        output_path (str): Name of output PDB file.
        n_repeats (int, optional): Number of pack-minimise rounds to perform. Larger
                                   values will significantly slow down the protocol.
                                   Defaults to 5.

    Returns:
        Pose: Minimised Pose
    """
    pose = pyrosetta.pose_from_pdb(input_path)
    # Set up the FastRelax protocol with specific repeats.
    relax = rosetta.protocols.relax.FastRelax(n_repeats)
    relax.set_scorefxn(pyrosetta.get_fa_scorefxn())  # Score using default function
    # Don't deviate too much from starting coords as Xtal structure provided.
    relax.constrain_relax_to_start_coords(True)
    # Run protocol and save output
    relax.apply(pose)
    pose.dump_pdb(output_path)

    return pose


def get_interface_energy(pose: Pose, interface: str) -> float:
    """Calculates the interface energy (DeltaG of binding) for a given complex. Briefly,
       the whole complex is scored using the default scoring function. Then, the
       individual components are separated and energies evaluated. The difference
       is the returned.

    Args:
        pose (Pose): Input PyRosetta pose object
        interface (str): Docking-style interface. For instance, to calculate the energy
                         between protein 1 (made up of chains A and B) and protein 2
                         (made up of chains X and Y), use AB_XY.

    Returns:
        float: DeltaG of binding for the complex.
    """

    # Initiate InterfaceAnalyzer and apply to pose object
    interface_analyser = rosetta.protocols.analysis.InterfaceAnalyzerMover(interface)
    interface_analyser.apply(pose)
    return interface_analyser.get_interface_dG()


def relax_local(
    pose: Pose, rosetta_residue_num: int, distance_threshold: float = 8.0
) -> Pose:
    """Locally minimises a structure around a given residue by moving both the backbone
        and side-chains.

    Args:
        pose (Pose): Input Pose object
        rosetta_residue_num (int): Index of residue in Pose around which miniisation
                                   should be performed. NB: This is likely to be
                                   different that the residue number in the PDB.
        distance_threshold (float, optional): The radius within which a residue must be
                                              to the residue of interest for it to be
                                              indcluded in the minimisation. Value in
                                              Angstroms. Defaults to 8.0.

    Returns:
        Pose: Locally minimised Pose object
    """

    # Setup the MoveMap to select which residues are included
    movemap = pyrosetta.MoveMap()
    movemap.set_bb(False)  # Initially set all backbone torsions to not move
    movemap.set_chi(False)  # Initially set all side chain torsions to not move

    # Identify neighbors and allow their movement
    # Since we are using CA distances here, we add 6A to the distance threshold as this
    # roughly ensures that residues with any heavy atom within the distance threshold move.
    for i in range(1, pose.total_residue() + 1):
        ca_dist = (
            pose.residue(rosetta_residue_num)
            .xyz("CA")
            .distance(pose.residue(i).xyz("CA"))
        )
        if (i == rosetta_residue_num) or (ca_dist < distance_threshold + 6):
            movemap.set_bb(i, True)
            movemap.set_chi(i, True)

    # Set up the Minimization with the standard scoring function and apply
    min_mover = pyrosetta.rosetta.protocols.minimization_packing.MinMover()
    min_mover.movemap(movemap)
    min_mover.score_function(pyrosetta.get_fa_scorefxn())
    min_mover.apply(pose)

    return pose


def mutate_and_relax(
    pose: Pose,
    output_file: str,
    rosetta_resnum: int,
    target_res: str,
    interface: str,
    distance_threshold: float = 8.0,
) -> float:
    """Mutates a residue and repacks side chains around it. Then relaxes neighbourhood
    main chain and side chains concurrently. Finally calculates the DeltaDeltaG of the
    mutation on the binding energy change. Also saves the resultant mutant struture as a
    PDB file.

    Args:
        pose (Pose): Input Pose object
        output_file (str): Name of output PDB file.
        rosetta_residue_num (int): Index of residue in Pose which will be mutated.
                                   NB: This is likely to be different that the residue
                                   number in the PDB file.
        target_res (str): One-letter code of target residue to which we want to mutate.
        interface (str): Docking-style interface. For instance, to calculate the energy
                         between protein 1 (made up of chains A and B) and protein 2
                         (made up of chains X and Y), use AB_XY.
        distance_threshold (float, optional): The radius within which a residue must be
                                              to the residue of interest for it to be
                                              indcluded in the minimisation. Value in
                                              Angstroms. Defaults to 8.0.

    Returns:
        float: Change in binding energy change (DeltaDeltaG) upon mutation.
    """

    # Get the initial binding energy change
    initial_energy = get_interface_energy(pose, interface)

    # Mutate target residue and repack surrounding side chains
    pyrosetta.toolbox.mutants.mutate_residue(
        pose,
        rosetta_resnum,
        target_res,
        pack_radius=distance_threshold,
        pack_scorefxn=None,
    )
    # Locally minimise the pose around the mutation, save the pose as a PDB file
    pose = relax_local(pose, rosetta_resnum, distance_threshold=distance_threshold)
    pose.dump_pdb(output_file)

    # Get the new binding energy change
    final_energy = get_interface_energy(pose, interface)

    return final_energy - initial_energy


def save_ddg(ddgs: np.ndarray, output_path: str, resnum: int, wt_res: str) -> None:
    """Plots the DeltaDeltaGs of a mutational scan for a position. Saves both the plot
       and the values as a npy file for later use.

    Args:
        ddgs (np.ndarray): (20) array of ddG values for each amino acid
                           (in alphabetical order based on one-letter codes)
        output_path (str): Path to output directory
        resnum (int): The residue number of that residue within the PDB file.
        wt_res (str): One-letter code of WT residue

    Returns:
        _type_: _description_
    """

    # Make and save plot
    f = plt.figure(figsize=[10, 5])
    plt.bar(x=AMINO_ACIDS, height=ddgs)
    plt.ylabel("$\Delta\Delta G$ (Reu.)")
    plt.xlabel("Amino Acid")
    plt.title(f"Mutants of {wt_res}{resnum}")
    f.savefig(os.path.join(output_path, "ddg_plot.pdf"), bbox_inches="tight")

    # Save data as npy file
    np.save(os.path.join(output_path, "ddgs.npy"), ddgs)

    return None


def mutate_and_relax_wrapper(args):
    """
    Wrapper function for multiprocessing to handle all arguments for mutate_and_relax.
    """
    pose, output_file, rosetta_resnum, res, interface, distance_threshold = args
    return mutate_and_relax(
        pose, output_file, rosetta_resnum, res, interface, distance_threshold
    )


def scan_residue_mutants(
    input_path: str,
    output_path: str,
    chain_id: str,
    resnum: int,
    wt_res: str,
    interface: str,
    distance_threshold: float = 8.0,
    n_cores: int = None,
) -> np.ndarray:
    """Performs a mutational scan for a given position in a protein. For each of the 20
       standard amino acids, the protein is loaded, mutated, and locally relaxed.
       The change in binding energy is then saved. To correct for the effect of the
       minimisation and mutation protcols, the change in binding energy upon "mutating"
       the residue to the WT is subtracted from all other mutations. Each mutated
       structure is then saved.

    Args:
        input_path (str): Path to input PDB file
        output_path (str): Path to output directory
        chain_id (str): Chain on which residue of interest is located in the PDB file.
        resnum (int): The residue number of that residue within the PDB file.
        wt_res (str): One-letter code of WT residue
        interface (str): Docking-style interface. For instance, to calculate the energy
                         between protein 1 (made up of chains A and B) and protein 2
                         (made up of chains X and Y), use AB_XY.
        distance_threshold (float, optional): The radius within which a residue must be
                                              to the residue of interest for it to be
                                              indcluded in the minimisation. Value in
                                              Angstroms. Defaults to 8.0.
        n_cores (int, optional): How many CPUs to utilise. If None, uses all available
                                 cores. Defaults to None.

    Returns:
        np.ndarray: (20) array of ddG values for each amino acid
                    (in alphabetical order based on one-letter codes)
    """

    # Make the output directory
    residue_path = os.path.join(output_path, f"{chain_id}_{resnum}")
    os.makedirs(residue_path, exist_ok=True)

    # Determine the number of processes
    if n_cores is None:
        n_cores = multiprocessing.cpu_count()

    # Create a multiprocessing pool
    with multiprocessing.Pool(n_cores) as pool:
        # Initialize empty list for arguments for each task
        tasks = []

        # Pre-load the structure to avoid reloading it in each process
        pose = pyrosetta.pose_from_pdb(input_path)
        pdb_info = pose.pdb_info()
        rosetta_resnum = pdb_info.pdb2pose(chain=chain_id, res=resnum)

        # Prepare tasks
        for res in AMINO_ACIDS:
            output_file = os.path.join(residue_path, f"{wt_res}{resnum}{res}.pdb")
            task = (
                pose.clone(),
                output_file,
                rosetta_resnum,
                res,
                interface,
                distance_threshold,
            )
            tasks.append(task)

        # Map tasks across the pool
        results = pool.map(mutate_and_relax_wrapper, tasks)

    # After collecting results, organize them into a ddgs array and apply correction
    ddgs = np.array(results)

    # Find the correction term for the WT residue and subtract from everything
    wt_index = AMINO_ACIDS.index(wt_res)
    ddg_correction = ddgs[wt_index]
    ddgs -= ddg_correction

    # Save the output
    save_ddg(ddgs, residue_path, resnum, wt_res)

    return ddgs
