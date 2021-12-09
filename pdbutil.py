import os
import argparse
import subprocess
from io import StringIO
from collections import namedtuple
from openmm import app
from openmm.app.pdbfile import PDBFile
from pdbfixer import PDBFixer


# constants for handle residues
RESPROT = ('ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS',
           'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP',
           'TYR', 'VAL', 'HID', 'HIE', 'HIN', 'HIP', 'CYX', 'ASH', 'GLH',
           'LYH', 'ACE', 'NME', 'GL4', 'AS4')
RESNA = ('C', 'G', 'U', 'A', 'DC', 'DG', 'DT', 'DA')
RESSOLV = ('WAT', 'HOH', 'AG', 'AL', 'Ag', 'BA', 'BR', 'Be', 'CA', 'CD', 'CE',
           'CL', 'CO', 'CR', 'CS', 'CU', 'CU1', 'Ce', 'Cl-', 'Cr', 'Dy', 'EU',
           'EU3', 'Er', 'F', 'FE', 'FE2', 'GD3', 'HE+', 'HG', 'HZ+', 'Hf',
           'IN', 'IOD', 'K', 'K+', 'LA', 'LI', 'LU', 'MG', 'MN', 'NA', 'NH4',
           'NI', 'Na+', 'Nd', 'PB', 'PD', 'PR', 'PT', 'Pu', 'RB', 'Ra', 'SM',
           'SR', 'Sm', 'Sn', 'SO4', 'TB', 'TL', 'Th', 'Tl', 'Tm', 'U4+', 'V2+', 'Y',
           'YB2', 'ZN', 'Zr')
AMBER_SUPPORT_RES = set(RESPROT + RESNA + RESSOLV)
AA_ATOMS = {
    'ALA': ('N', 'H', 'CA', 'HA', 'CB', 'HB1', 'HB2', 'HB3', 'C', 'O'),
    'ARG': ('N', 'H', 'CA', 'HA', 'CB', 'HB2', 'HB3', 'CG', 'HG2', 'HG3', 'CD', 'HD2', 'HD3', 'NE', 'HE', 'CZ',
            'NH1', 'HH11', 'HH12', 'NH2', 'HH21', 'HH22', 'C', 'O'),
    'ASH': ('N', 'H', 'CA', 'HA', 'CB', 'HB2', 'HB3', 'CG', 'OD1', 'OD2', 'HD2', 'C', 'O'),
    'ASN': ('N', 'H', 'CA', 'HA', 'CB', 'HB2', 'HB3', 'CG', 'OD1', 'ND2', 'HD21', 'HD22', 'C', 'O'),
    'ASP': ('N', 'H', 'CA', 'HA', 'CB', 'HB2', 'HB3', 'CG', 'OD1', 'OD2', 'C', 'O'),
    'CYM': ('N', 'H', 'CA', 'HA', 'CB', 'HB3', 'HB2', 'SG', 'C', 'O'),
    'CYS': ('N', 'H', 'CA', 'HA', 'CB', 'HB2', 'HB3', 'SG', 'HG', 'C', 'O'),
    'CYX': ('N', 'H', 'CA', 'HA', 'CB', 'HB2', 'HB3', 'SG', 'C', 'O'),
    'GLH': ('N', 'H', 'CA', 'HA', 'CB', 'HB2', 'HB3', 'CG', 'HG2', 'HG3', 'CD', 'OE1', 'OE2', 'HE2', 'C', 'O'),
    'GLN': ('N', 'H', 'CA', 'HA', 'CB', 'HB2', 'HB3', 'CG', 'HG2', 'HG3', 'CD', 'OE1', 'NE2', 'HE21', 'HE22', 'C', 'O'),
    'GLU': ('N', 'H', 'CA', 'HA', 'CB', 'HB2', 'HB3', 'CG', 'HG2', 'HG3', 'CD', 'OE1', 'OE2', 'C', 'O'),
    'GLY': ('N', 'H', 'CA', 'HA2', 'HA3', 'C', 'O'),
    'HID': ('N', 'H', 'CA', 'HA', 'CB', 'HB2', 'HB3', 'CG', 'ND1', 'HD1', 'CE1', 'HE1', 'NE2', 'CD2', 'HD2', 'C', 'O'),
    'HIE': ('N', 'H', 'CA', 'HA', 'CB', 'HB2', 'HB3', 'CG', 'ND1', 'CE1', 'HE1', 'NE2', 'HE2', 'CD2', 'HD2', 'C', 'O'),
    'HIP': ('N', 'H', 'CA', 'HA', 'CB', 'HB2', 'HB3', 'CG', 'ND1', 'HD1', 'CE1', 'HE1', 'NE2', 'HE2', 'CD2', 'HD2',
            'C', 'O'),
    'HYP': ('N', 'CD', 'HD22', 'HD23', 'CG', 'HG', 'OD1', 'HD1', 'CB', 'HB2', 'HB3', 'CA', 'HA', 'C', 'O'),
    'ILE': ('N', 'H', 'CA', 'HA', 'CB', 'HB', 'CG2', 'HG21', 'HG22', 'HG23', 'CG1', 'HG12', 'HG13', 'CD1', 'HD11',
            'HD12', 'HD13', 'C', 'O'),
    'LEU': ('N', 'H', 'CA', 'HA', 'CB', 'HB2', 'HB3', 'CG', 'HG', 'CD1', 'HD11', 'HD12', 'HD13', 'CD2', 'HD21', 'HD22',
            'HD23', 'C', 'O'),
    'LYN': ('N', 'H', 'CA', 'HA', 'CB', 'HB2', 'HB3', 'CG', 'HG2', 'HG3', 'CD', 'HD2', 'HD3', 'CE', 'HE2', 'HE3', 'NZ',
            'HZ2', 'HZ3', 'C', 'O'),
    'LYS': ('N', 'H', 'CA', 'HA', 'CB', 'HB2', 'HB3', 'CG', 'HG2', 'HG3', 'CD', 'HD2', 'HD3', 'CE', 'HE2', 'HE3', 'NZ',
            'HZ1', 'HZ2', 'HZ3', 'C', 'O'),
    'MET': ('N', 'H', 'CA', 'HA', 'CB', 'HB2', 'HB3', 'CG', 'HG2', 'HG3', 'SD', 'CE', 'HE1', 'HE2', 'HE3', 'C', 'O'),
    'PHE': ('N', 'H', 'CA', 'HA', 'CB', 'HB2', 'HB3', 'CG', 'CD1', 'HD1', 'CE1', 'HE1', 'CZ', 'HZ', 'CE2', 'HE2', 'CD2',
            'HD2', 'C', 'O'),
    'PRO': ('N', 'CD', 'HD2', 'HD3', 'CG', 'HG2', 'HG3', 'CB', 'HB2', 'HB3', 'CA', 'HA', 'C', 'O'),
    'SER': ('N', 'H', 'CA', 'HA', 'CB', 'HB2', 'HB3', 'OG', 'HG', 'C', 'O'),
    'THR': ('N', 'H', 'CA', 'HA', 'CB', 'HB', 'CG2', 'HG21', 'HG22', 'HG23', 'OG1', 'HG1', 'C', 'O'),
    'TRP': ('N', 'H', 'CA', 'HA', 'CB', 'HB2', 'HB3', 'CG', 'CD1', 'HD1', 'NE1', 'HE1', 'CE2', 'CZ2', 'HZ2', 'CH2',
            'HH2', 'CZ3', 'HZ3', 'CE3', 'HE3', 'CD2', 'C', 'O'),
    'TYR': ('N', 'H', 'CA', 'HA', 'CB', 'HB2', 'HB3', 'CG', 'CD1', 'HD1', 'CE1', 'HE1', 'CZ', 'OH', 'HH', 'CE2',
            'HE2', 'CD2', 'HD2', 'C', 'O'),
    'VAL': ('N', 'H', 'CA', 'HA', 'CB', 'HB', 'CG1', 'HG11', 'HG12', 'HG13', 'CG2', 'HG21', 'HG22', 'HG23', 'C', 'O'),
}
MUTMAP = {
    'SEP': 'SER',
    'TPO': 'THR',
    'PTR': 'TYR',
    'PDS': 'ASP',
    'PHL': 'ASP',
    'MLY': 'LYS',
    'CSP': 'CYS',
    'MSE': 'MET',
    'GMA': 'GLU',
    'OCS': 'CYS',
    'CSX': 'CYS',
}


def get_force_field(ff_name='ff99SBildn', water_model_name=None):
    if ff_name == 'ff99SBildn':
        ff_file_name = 'amber99sbildn.xml'
        if water_model_name == 'TIP3PBOX':
            return app.ForceField(ff_file_name, 'tip3p.xml')
        elif water_model_name == 'SPCEBOX':
            return app.ForceField(ff_file_name, 'spce.xml')
        elif water_model_name == 'TIP4PEWBOX':
            return app.ForceField(ff_file_name, 'tip4pew.xml')
        elif water_model_name is None:
            return app.ForceField(ff_file_name)
        else:
            raise ValueError('%s Water model not found for force field %s' % (water_model_name, ff_name))
    elif ff_name == 'ff14SB':
        ff_file_name = 'amber14/protein.ff14SB.xml'
        if water_model_name == 'TIP3PBOX':
            return app.ForceField(ff_file_name, 'amber14/tip3p.xml')
        elif water_model_name == 'SPCEBOX':
            return app.ForceField(ff_file_name, 'amber14/spce.xml')
        elif water_model_name == 'TIP4PEWBOX':
            return app.ForceField(ff_file_name, 'amber14/tip4pew.xml')
        elif water_model_name is None:
            return app.ForceField(ff_file_name)
        else:
            raise ValueError('%s Water model not found for force field %s' % (water_model_name, ff_name))
    elif ff_name == 'CHARMM36':
        ff_file_name = 'charm36.xml'
        if water_model_name == 'TIP3PBOX':
            return app.ForceField(ff_file_name, 'charm36/water.xml')
        elif water_model_name == 'SPCEBOX':
            return app.ForceField(ff_file_name, 'charm36/spce.xml')
        elif water_model_name == 'TIP4PEWBOX':
            return app.ForceField(ff_file_name, 'charm36/tip4pew.xml')
        elif water_model_name == 'TIP5PBOX':
            return app.ForceField(ff_file_name, 'charm36/tip5p.xml')
        elif water_model_name is None:
            return app.ForceField(ff_file_name)
        else:
            raise ValueError('%s Water model not found for force field %s' % (water_model_name, ff_name))


AtomRecord = namedtuple('AtomRecord', ['name', 'atom_number', 'atom_name', 'altloc', 'residue_name', 'chain_id',
                                       'residue_number', 'icode', 'x', 'y', 'z', 'occupancy', 'temp_factor', 'element'])
TerminalRecord = namedtuple('TerminalRecord', ['name', 'atom_number', 'residue_name', 'chain_id', 'residue_number',
                                               'icode'])
ConnectRecord = namedtuple('ConnectRecord', ['name', 'serial0', 'serial1', 'serial2', 'serial3', 'serial4'])
SeqresRecord = namedtuple('SeqresRecord', ['name', 'serial', 'chain_id', 'num_residues', 'resname0', 'resname1',
                                           'resname2', 'resname3', 'resname4', 'rename5', 'resname6', 'resname7',
                                           'resname8', 'resname9', 'resname10', 'resname11', 'resname12'])


def parse_record(line):
    _record_name = line[:6].strip()

    if _record_name in ('ATOM', 'HETATM'):
        _atom_number = int(line[6:11].strip())
        _atom_name = line[12:16].strip()
        _altloc = line[16].strip()
        _residue_name = line[17:20].strip()
        _chain_id = line[21]
        _residue_number = int(line[22:26].strip())
        _icode = line[26].strip()
        _x = float(line[30:38].strip())
        _y = float(line[38:46].strip())
        _z = float(line[46:54].strip())
        _occupancy = float(line[54:60].strip() or 0.0)
        _temp_factor = float(line[60:66].strip() or 0.0)
        _element = line[76:78].strip()
        return AtomRecord(_record_name, _atom_number, _atom_name, _altloc, _residue_name, _chain_id, _residue_number,
                          _icode, _x, _y, _z, _occupancy, _temp_factor, _element)
    elif _record_name == 'TER':
        _atom_number = int(line[6:11].strip())
        _residue_name = line[17:20].strip()
        _chain_id = line[21]
        _residue_number = int(line[22:26].strip())
        _icode = line[26].strip()
        return TerminalRecord(_record_name, _atom_number, _residue_name, _chain_id, _residue_number, _icode)
    elif _record_name == 'CONECT':
        _serial0 = line[6:11].strip()
        _serial1 = line[11:16].strip()
        _serial2 = line[16:21].strip()
        _serial3 = line[21:26].strip()
        _serial4 = line[26:31].strip()
        return ConnectRecord(_record_name, _serial0, _serial1, _serial2, _serial3, _serial4)
    elif _record_name == 'MODEL':
        return namedtuple('ModelRecord', 'name, serial')(_record_name, int(line[10:14].strip()))
    elif _record_name == 'ENDMDL':
        return namedtuple('EndModelRecord', 'name')(_record_name)
    else:
        raise ValueError('Invalid string.')


def format_record(record):
    if record.name in ('ATOM', 'HETATM'):
        return '%-6s%5d %-4s%1s%3s %1s%4d%1s   %8.3f%8.3f%8.3f%6.2f%6.2f          %2s' % record
    elif record.name == 'TER':
        return '%-6s%5d      %3s %1s%4d%1s' % record
    elif record.name == 'CONECT':
        return '%-6s%5s%5s%5s%5s%5s' % record
    elif record.name == 'MODEL':
        return '%-6s    %4d' % record
    elif record.name == 'ENDMDL':
        return record.name
    else:
        raise ValueError('Invalid record.')


class PDBObject:
    def __init__(self, f):
        self.content = []
        for line in f:
            try:
                self.content.append(parse_record(line))
            except ValueError:
                pass

    @classmethod
    def from_pdb_file(cls, f):
        instance = cls([])

        for line in f:
            try:
                instance.content.append(parse_record(line))
            except ValueError:
                pass
        return instance

    @property
    def pdb_string(self):
        lines = []
        for record in self.content:
            try:
                lines.append(format_record(record))
            except ValueError:
                pass

        return '\n'.join(lines)

    @property
    def pdbfixer(self):
        with StringIO('\n'.join([format_record(record) for record in self.content])) as f:
            return PDBFixer(pdbfile=f)

    def get_pdbfixer(self):
        return self.pdbfixer

    @classmethod
    def load(cls, topology, positions):
        with StringIO() as f:
            app.PDBFile.writeFile(topology, positions, f)
            f.seek(0)
            return cls(f)

    # def load(self, topology, positions):
    #     with StringIO() as f:
    #         PDBFile.writeFile(topology, positions, f)
    #         f.seek(0)
    #         self.content = []
    #         for line in f:
    #             try:
    #                 self.content.append(parse_record(line))
    #             except ValueError:
    #                 pass


def get_chains(pdb_file: os.PathLike):
    with open(pdb_file, 'r') as f:
        pdb_object = PDBObject(f)

    _chains = set()
    for record in pdb_object.content:
        if record.name in ('ATOM', 'HETATM', 'TER'):
            _chains.add(record.chain_id)
    return sorted(_chains)


def get_models(pdb_file: os.PathLike):
    with open(pdb_file, 'r') as f:
        pdb_object = PDBObject(f)

    _models = {1}
    for record in pdb_object.content:
        if record.name == 'MODEL':
            _models.add(record.serial)
    return sorted(_models)


def get_heterogenes(pdb_file: os.PathLike):
    with open(pdb_file, 'r') as f:
        pdb_object = PDBObject(f)

    ret = set()
    for record in pdb_object.content:
        if record.name != 'HETATM':
            continue

        if record.residue_name in RESSOLV or record.residue_name in MUTMAP:
            continue

        ret.add((record.chain_id, record.residue_number, record.residue_name))
    return sorted(ret)


def select_model_and_chains(pdb_file: os.PathLike, selected_model: int, selected_chains: list):
    def select_model(_pdb_content: list, _selected_model: int):
        _selected = []
        adding = True
        for _record in _pdb_content:
            if _record.name == 'MODEL' and _record.serial == _selected_model:
                adding = True
                continue
            if _record.name == 'MODEL' and _record.serial != _selected_model:
                adding = False
                continue
            if _record.name == 'ENDMDL':
                continue

            if adding:
                _selected.append(_record)
        return _selected

    def select_chains(_pdb_content: list, _selected_chains: list):
        _selected = []
        for _record in _pdb_content:
            # selecte chain
            if _record.chain_id not in _selected_chains:
                continue
            # add SEQRES
            if _record.name == 'SEQRES':
                _selected.append(_record)
                continue
            # strip water
            if _record.residue_name in ('WAT', 'HOH'):
                continue
            # add ter
            if _record.name == 'TER':
                _selected.append(_record)
                continue
            # remove hydrogens
            if _record.atom_name[0] == 'H':
                continue
            # remove hydrogens
            if _record.element in ('H', 'D'):
                continue
            _selected.append(_record)
        return _selected

    with open(pdb_file, 'r') as f:
        pdb_object = PDBObject(f)

    pdb_object.content = select_chains(select_model(pdb_object.content, selected_model), selected_chains)
    with open(pdb_file, 'w') as f:
        f.write(pdb_object.pdb_string)


def cleanup(pdb_file: os.PathLike):
    def process_altloc(_pdb_content):
        _selected = []
        for _record in _pdb_content:
            if _record.name in ('ATOM', 'HETATM') and _record.altloc and _record.altloc != 'A':
                continue
            _selected.append(_record)
        return _selected

    def process_icode(_pdb_content):
        _icode_residues = dict()

        for _record in _pdb_content:
            if _record.name not in ('ATOM', 'HETATM', 'TER'):
                continue
            key = _record.chain_id, _record.residue_number
            if key not in _icode_residues:
                _icode_residues[key] = set()
                _icode_residues[key].add(_record.icode)

        _selected = []
        for _record in _pdb_content:
            if _record.name in ('ATOM', 'HETATM', 'TER'):
                key = _record.chain_id, _record.residue_number
                if len(_icode_residues[key]) > 1 and sorted(_icode_residues[key])[0] != _record.icode:
                    continue
            _selected.append(_record)
        return _selected

    def process_hetero_residues(_pdb_content):
        hetero_residues = get_heterogenes(pdb_file)
        _selected = []
        for _record in _pdb_content:
            if _record.name == 'HETATM' and (
                    _record.chain_id, _record.residue_number, _record.residue_name) in hetero_residues:
                continue
            _selected.append(_record)
        return _selected

    with open(pdb_file, 'r') as f:
        pdb_object = PDBObject(f)

    pdb_object.content = process_hetero_residues(process_icode(process_altloc(pdb_object.content)))
    with open(pdb_file, 'w') as f:
        f.write(pdb_object.pdb_string)


def add_missing_atoms(pdb_file: os.PathLike):
    with open(pdb_file, 'r') as f:
        pdb_object = PDBObject(f)

    pf = pdb_object.get_pdbfixer()
    pf.findMissingResidues()
    chain_len = [len(list(chain.residues())) for chain in pf.topology.chains()]

    # remove N or C terminal missing
    del_keys = []
    int_keys = []  # internal missing residue keys
    for mr in pf.missingResidues:
        # N-term or C-term
        if mr[1] == 0 or chain_len[mr[0]] == mr[1]:
            del_keys.append(mr)
        else:
            int_keys.append(mr)

    for dk in del_keys:
        del pf.missingResidues[dk]

    chains = get_chains(pdb_file)
    replaced_res = []
    for ik in int_keys:
        for index, res in enumerate(pf.missingResidues[ik]):
            if res in AMBER_SUPPORT_RES:
                continue

            if res in MUTMAP:
                replaced_res.append((chains[ik[0]], ik[1] + index + 1, pf.missingResidues[ik][index]))
                pf.missingResidues[ik][index] = MUTMAP[res]
            else:
                replaced_res.append((chains[ik[0]], ik[1] + index + 1, pf.missingResidues[ik][index]))
                pf.missingResidues[ik][index] = 'GLY'

    # to aviod template not found error on non standard missing residue
    # change missing residue names to standard amino acids
    # and add missing atoms then change name again its original
    pf.findMissingAtoms()
    pf.addMissingAtoms()
    pdb_object = PDBObject.load(pf.topology, pf.positions)
    # Missing Residue가 Non standard인 경우 GLY로 치환하는데 그것을 다시 원래 Residue 이름으로 바꿔서 User가 선택할 수 있도록 함
    if replaced_res:
        res_map = dict()
        for chain, resnum, new_resname in replaced_res:
            res_map[(chain, resnum)] = new_resname

        _selected = []
        for record in pdb_object.content:
            if record.name in ('ATOM', 'HETATM') and (record.chain_id, record.residue_number) in res_map:
                record.residue_name = res_map[(record.chain_id, record.residue_number)]
            _selected.append(record)
        pdb_object.content = _selected

    with open(pdb_file, 'w') as f:
        f.write(pdb_object.pdb_string)


def get_non_standard_residues(pdb_file: os.PathLike):
    with open(pdb_file, 'r') as f:
        pdb_object = PDBObject(f)

    pf = pdb_object.get_pdbfixer()
    pf.findNonstandardResidues()
    return pf.nonstandardResidues


def get_disulfide_bond_candidates(pdb_file: os.PathLike):
    import numpy as np
    from scipy.spatial import distance
    from simtk import unit
    with open(pdb_file, 'r') as f:
        pdb_object = PDBObject(f)

    pf = pdb_object.get_pdbfixer()
    candidates = []
    for chain in pf.topology.chains():
        cys_residues = []
        coordinates = []
        for residue in chain.residues():
            if residue.name in ('CYS', 'CYX'):
                cys_residues.append(residue)
                for atom in residue.atoms():
                    if atom.name == 'SG':
                        coordinates.append(pf.positions[atom.index].value_in_unit(unit.nano * unit.meter))

        if not coordinates:
            continue

        dist = distance.pdist(np.array(coordinates), metric='euclidean')

        idist = 0
        for i, r1 in enumerate(cys_residues):
            for j, r2 in enumerate(cys_residues):
                if i >= j:
                    continue

                d = dist[idist]
                idist += 1

                if d < 0.3:
                    candidates.append((chain.id, r1, r2, d * 10))

    return candidates


def get_solvent_ions(pdb_file: os.PathLike):
    with open(pdb_file, 'r') as f:
        pdb_object = PDBObject(f)

    ret = set()
    for record in pdb_object.content:
        if record.name not in ('ATOM', 'HETATM'):
            continue

        if record.residue_name in RESSOLV:
            ret.add((record.chain_id, record.residue_name))

    return sorted(ret)


def get_unkown_protonation_states(pdb_file: os.PathLike):
    with open(pdb_file, 'r') as f:
        pdb_object = PDBObject(f)

    pf = pdb_object.get_pdbfixer()
    histidines = []
    for residue in pf.topology.residues():
        if residue.name in ('HIS', 'HID', 'HIE', 'HIP', 'HIN'):
            histidines.append(residue)
    return histidines


def convert_non_standard_residues(pdb_file: os.PathLike):
    with open(pdb_file, 'r') as f:
        pdb_object = PDBObject(f)

    pf = pdb_object.get_pdbfixer()
    pf.missingResidues = {}
    pf.findNonstandardResidues()
    pf.replaceNonstandardResidues()
    pf.findMissingAtoms()
    pf.addMissingAtoms()

    pdb_object = PDBObject.load(pf.topology, pf.positions)
    with open(pdb_file, 'w') as f:
        f.write(pdb_object.pdb_string)


def determine_protonation_states(pdb_file: os.PathLike, protonation_states: dict):
    with open(pdb_file, 'r') as f:
        pdb_object = PDBObject(f)

    pf = pdb_object.get_pdbfixer()

    variants = []
    for chain in pf.topology.chains():
        for residue in chain.residues():
            if (chain.id, residue.id) in protonation_states:
                variant = protonation_states[(chain.id, residue.id)]
                if variant != 'HIS':
                    variants.append(variant)
                    continue
            variants.append(None)
    return variants


def keep_solvent_ions(pdb_file: os.PathLike, selected_ions: list):
    with open(pdb_file, 'r') as f:
        pdb_object = PDBObject(f)

    solvent_ions = get_solvent_ions(pdb_file)

    _selected = []
    for _record in pdb_object.content:
        if _record.name in ('ATOM', 'HETATM') and (_record.chain_id, _record.residue_name) in solvent_ions and (
                _record.chain_id, _record.residue_name) not in selected_ions:
            continue
        _selected.append(_record)

    pdb_object.content = _selected
    with open(pdb_file, 'w') as f:
        f.write(pdb_object.pdb_string)


def add_missing_hydrogens(pdb_file: os.PathLike, variants: list, forcefield='ff99SBildn'):
    with open(pdb_file, 'r') as f:
        pdb_object = PDBObject(f)

    pf = pdb_object.get_pdbfixer()
    ff = get_force_field(forcefield, water_model_name=None)
    mdl = app.Modeller(pf.topology, pf.positions)
    mdl.addHydrogens(forcefield=ff, variants=variants or determine_protonation_states(pdb_file, dict()))

    pdb_object = PDBObject.load(mdl.topology, mdl.positions)
    with open(pdb_file, 'w') as f:
        f.write(pdb_object.pdb_string)


def auto_pre_process(pdb_file: os.PathLike, selected_chains: list = None, forcefield='ff99SBildn'):
    if not selected_chains:
        selected_chains = get_chains(pdb_file)

    models = get_models(pdb_file)

    select_model_and_chains(pdb_file, models[0], selected_chains)
    cleanup(pdb_file)
    convert_non_standard_residues(pdb_file)
    keep_solvent_ions(pdb_file, [])
    variants = determine_protonation_states(pdb_file, dict())
    add_missing_hydrogens(pdb_file, variants, forcefield=forcefield)


def submit_batch(func, base_dir=os.getcwd(), partition='prowave', **kwargs):
    stdout = os.path.join(base_dir, 'stdout')
    stderr = os.path.join(base_dir, 'stderr')

    sbatch_cmd = os.path.join(os.environ.get('SLURM_HOME', '/usr/local'), 'bin/sbatch')

    cmd = [
        sbatch_cmd,
        '--output', stdout,
        '--error', stderr,
        '--nodes', '1',
        '--time', '72:00:00',
        '--job-name', 'PREPROC',
        '--ntasks', '1',
        '--partition', partition,
        os.path.abspath(__file__), func, '--pdb_file', os.path.abspath(kwargs['pdb_file']),
    ]

    if func == 'auto':
        if 'forcefield' in kwargs:
            cmd += ['--forcefield', kwargs['forcefield']]
        if 'chains' in kwargs:
            cmd += ['--chains'] + kwargs['chains']
    elif func == 'add_missing_atoms':
        pass
    elif func == 'add_missing_hydrogens':
        if 'forcefield' in kwargs:
            cmd += ['--forcefield', kwargs['forcefield']]
        if 'variants' in kwargs:
            cmd += '--variants' + kwargs['variants']
    else:
        raise ValueError('Wrong option')

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()

    submitted_msg = out.decode()
    error_msg = err.decode()

    if error_msg:
        print(error_msg)

    if submitted_msg.startswith('Submitted batch job'):
        batch_jobid = int(submitted_msg.split()[3])

        with open(os.path.join(base_dir, 'status'), 'w') as f:
            f.write('submitted %d' % batch_jobid)

        return batch_jobid
    else:
        return '-1'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='func')
    parser_auto = subparsers.add_parser('auto')
    parser_auto.add_argument('--pdb_file', default=os.path.join(os.getcwd(), 'model.pdb'))
    parser_auto.add_argument('--forcefield', default='ff99SBildn')
    parser_auto.add_argument('--chains', nargs='+')

    parser_ama = subparsers.add_parser('add_missing_atoms')
    parser_ama.add_argument('--pdb_file', default=os.path.join(os.getcwd(), 'model.pdb'))

    parser_amh = subparsers.add_parser('add_missing_hydrogens')
    parser_amh.add_argument('--pdb_file', default=os.path.join(os.getcwd(), 'model.pdb'))
    parser_amh.add_argument('--forcefield', default='ff99SBildn')
    parser_amh.add_argument('--variants', nargs='+')

    args = parser.parse_args()

    if args.func == 'auto':
        auto_pre_process(args.pdb_file, args.chains, args.forcefield)

    elif args.func == 'add_missing_atoms':
        add_missing_atoms(args.pdb_file)

    elif args.func == 'add_missing_hydrogens':
        add_missing_hydrogens(args.pdb_file, variants=args.variants, forcefield=args.forcefield)
