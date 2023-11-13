#!/usr/bin/env python

import pathlib as P
prj_root = P.Path(__file__).parent.parent
import sys
if (p := str(prj_root)) not in sys.path:
    sys.path.append(p)

from preprocessing.create_nrPDB_GO_annot import read_fasta, load_clusters
from preprocessing.biotoolbox.structure_file_reader import build_structure_container_for_pdb
from preprocessing.biotoolbox.contact_map_builder import DistanceMapBuilder
from Bio.PDB import PDBList

from functools import partial
import numpy as np
import argparse
import csv
import os
import re
import gzip

SUFFIXPATTERN = re.compile(r'(?!^)\.[^\.]+$')

# ALPHAFOLDNAME = re.compile(r'(?:AF-(\w+)-F1-model_v4(?:\.[^.]+)+$)')

def make_distance_maps(pdbfile, chain=None, sequence=None):
    """
    Generate (diagonalized) C_alpha and C_beta distance matrix from a pdbfile
    """
    # gzip format and pdb (cif) format
    m = SUFFIXPATTERN.search(pdbfile)
    if m is None:
        raise ValueError(f"Unknown file format {pdbfile}")
    elif (r := m.group()) in [".pdb", ".cif"]:
            pdb_handle = open(pdbfile, 'r')
    elif r in [".gz", ".gzip"]:
            pdb_handle = gzip.open(pdbfile, "rt")
    else:
        raise ValueError(f"Unknown file format {pdbfile}")
    structure_container = build_structure_container_for_pdb(pdb_handle.read(), chain).with_seqres(sequence)
    # structure_container.chains = {chain: structure_container.chains[chain]}

    mapper = DistanceMapBuilder(atom="CA", glycine_hack=-1)  # start with CA distances
    ca = mapper.generate_map_for_pdb(structure_container)
    cb = mapper.set_atom("CB").generate_map_for_pdb(structure_container)
    pdb_handle.close()

    return ca.chains, cb.chains


def load_GO_annot(filename):
    """ Load GO annotations """
    onts = ['molecular_function', 'biological_process', 'cellular_component']
    prot2annot = {}
    goterms = {ont: [] for ont in onts}
    gonames = {ont: [] for ont in onts}
    with open(filename, mode='r') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')

        # molecular function
        next(reader, None)  # skip the headers
        goterms[onts[0]] = next(reader)
        next(reader, None)  # skip the headers
        gonames[onts[0]] = next(reader)

        # biological process
        next(reader, None)  # skip the headers
        goterms[onts[1]] = next(reader)
        next(reader, None)  # skip the headers
        gonames[onts[1]] = next(reader)

        # cellular component
        next(reader, None)  # skip the headers
        goterms[onts[2]] = next(reader)
        next(reader, None)  # skip the headers
        gonames[onts[2]] = next(reader)

        next(reader, None)  # skip the headers
        for row in reader:
            prot, prot_goterms = row[0], row[1:]
            prot2annot[prot] = {ont: [] for ont in onts}
            for i in range(3):
                prot2annot[prot][onts[i]] = [goterm for goterm in prot_goterms[i].split(',') if goterm != '']
    return prot2annot, goterms, gonames


def load_EC_annot(filename):
    """ Load EC annotations """
    prot2annot = {}
    ec_numbers = []
    with open(filename, mode='r') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')

        # molecular function
        next(reader, None)  # skip the headers
        ec_numbers = next(reader)
        next(reader, None)  # skip the headers
        for row in reader:
            prot, prot_ec_numbers = row[0], row[1]
            prot2annot[prot] = [ec_num for ec_num in prot_ec_numbers.split(',')]
    return prot2annot, ec_numbers


def pdb2distMap(pdb, chain, chain_seqres, pdir,
                 **kwargs):
    no_retrieve = kwargs.get('no_retrieve', False)
    gzip = kwargs.get('gzip', False)
    if not no_retrieve:
        pdb_list = PDBList()
        pdb_list.retrieve_pdb_file(pdb, pdir=pdir)
    ca, cb = make_distance_maps(pdir + '/' + pdb +'.cif' + ('.gz' if gzip else ''), 
                                chain=chain, sequence=chain_seqres)

    return ca[chain]['contact-map'], cb[chain]['contact-map']


def load_list(fname):
    """
    Load PDB chains
    """
    pdb_chain_list = set()
    fRead = open(fname, 'r')
    for line in fRead:
        pdb_chain_list.add(line.strip())
    fRead.close()

    return pdb_chain_list


def write_annot_npz(prot, prot2seq=None,in_dir=None, out_dir=None,
                    **kwargs):
    """
    Write to *.npz file format.
    """
    alphafold_db = kwargs.get('alphafold_db', False)
    if alphafold_db:
        # pdb = ALPHAFOLDNAME.search(prot).group(1)
        # m = ALPHAFOLDNAME.search(prot)
        # assert m is not None, f"Cannot parse {prot}"
        # pdb = m.group(1)
        pdb = prot
        chain = "A" # default chain is A
    else:
        pdb, chain = prot.split('-')
    print ('pdb=', pdb, 'chain=', chain)
    pdir = in_dir if in_dir is not None else os.path.join(out_dir, 'tmp_PDB_files_dir')
    try:
        A_ca, A_cb = pdb2distMap(pdb.lower() if not alphafold_db else pdb,
                                 chain, 
                                 prot2seq[prot] if not alphafold_db else prot2seq[pdb], 
                                 pdir=pdir,
                                 **kwargs)
        np.savez_compressed(os.path.join(out_dir, prot),
                            C_alpha=A_ca,
                            C_beta=A_cb,
                            seqres=prot2seq[prot],
                            )
    except Exception as e:
        print (e)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-annot', type=str, help="Input file (*.tsv) with preprocessed annotations.")
    parser.add_argument('-ec', help="Use EC annotations.", action="store_true")
    parser.add_argument('-seqres', type=str, default='./data/pdb_seqres.txt.gz', help="PDB chain seqres fasta.")
    parser.add_argument('-num_threads', type=int, default=20, help="Number of threads (CPUs) to use in the computation.")
    parser.add_argument('-bc', type=str, help="Clusters of PDB chains computd by Blastclust.")
    parser.add_argument("-in_dir", type=str, default=None, help="Input directory with PDB files.")
    parser.add_argument('-out_dir', type=str, default='./data/annot_pdb_chains_npz/', help="Output directory with distance maps saved in *.npz format.")
    parser.add_argument("--no_retrieve", action="store_true", help="Do not retrieve PDB files.")
    parser.add_argument("--alphafold_db", action="store_true", help="from alphafold_db")
    parser.add_argument("--gzip", action="store_true", help="PDB files are in gzip format.")
    args = parser.parse_args()

    # load annotations
    prot2goterms = {}
    if args.annot is not None:
        if args.ec:
            prot2goterms, _ = load_EC_annot(args.annot)
        else:
            prot2goterms, _, _ = load_GO_annot(args.annot)
        print ("### number of annotated proteins: %d" % (len(prot2goterms)))

    # load sequences
    prot2seq = read_fasta(args.seqres)
    print ("### number of proteins with seqres sequences: %d" % (len(prot2seq)))

    # load clusters
    pdb2clust = {}
    if args.bc is not None:
        pdb2clust = load_clusters(args.bc)
        clusters = set([pdb2clust[prot][0] for prot in prot2goterms])
        print ("### number of annotated clusters: %d" % (len(clusters)))

    """
    # extracting unannotated proteins
    unannot_prots = set()
    for prot in pdb2clust:
        if (pdb2clust[prot][0] not in clusters) and (pdb2clust[prot][1] == 0) and (prot in prot2seq):
            unannot_prots.add(prot)
    print ("### number of unannot proteins: %d" % (len(unannot_prots)))
    """

    to_be_processed = set(prot2seq.keys())
    if len(prot2goterms) != 0:
        to_be_processed = to_be_processed.intersection(set(prot2goterms.keys()))
    if pdb2clust != {} and len(prot2goterms) != 0:
        to_be_processed = to_be_processed.intersection(set(pdb2clust.keys()))
    print ("Number of pdbs to be processed=", len(to_be_processed))
    print (to_be_processed)

    # process on multiple cpus
    nprocs = args.num_threads
    in_dir = args.in_dir
    out_dir = args.out_dir
    import multiprocessing
    nprocs = np.minimum(nprocs, multiprocessing.cpu_count())
    if nprocs > 4:
        pool = multiprocessing.Pool(processes=nprocs)
        pool.map(partial(write_annot_npz, 
                         prot2seq=prot2seq, 
                         in_dir=in_dir, 
                         out_dir=out_dir,
                         no_retrieve=args.no_retrieve,
                         alphafold_db=args.alphafold_db,
                         gzip=args.gzip,),
                 to_be_processed,
                 )
    else:
        for prot in to_be_processed:
            write_annot_npz(prot, prot2seq=prot2seq, out_dir=out_dir)
