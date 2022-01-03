#!/home/nbcc/anaconda3/envs/prowave/bin/python
"""
SM Molecular Dynamics simulation package

since 2018. 11. 21.
Junwon Lee
esanzy87@gmail.com
"""
import os
import argparse
import gzip
import json
import tempfile
import shutil
import subprocess
import numpy as np
import mdtraj as mdt
from simtk import openmm as mm, unit
from simtk.openmm import app


def get_platform():
    nplatforms = mm.Platform.getNumPlatforms()

    platform = None
    for i in reversed(range(nplatforms)):
        platform = mm.Platform.getPlatform(i)
        if platform.getName() == 'CUDA':
            break
        elif platform.getName() == 'OpenCL':
            break

    return platform


def get_restrainted_force(restraint_weight, pdb):
    _res_weight = restraint_weight * 418.4  # kcal/mol/A^2 to kJ/mol/nm^2

    force = mm.CustomExternalForce('k*(periodicdistance(x,y,z,x0,y0,z0)^2)')
    force.addGlobalParameter('k', _res_weight)
    force.addPerParticleParameter('x0')
    force.addPerParticleParameter('y0')
    force.addPerParticleParameter('z0')

    for atom in pdb.topology.atoms():
        resname = atom.residue.name

        if resname not in ('HOH', 'NA', 'CL'):
            force.addParticle(atom.index, pdb.positions[atom.index])
    return force


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
        ff_file_name = 'charmm36.xml'
        if water_model_name == 'TIP3PBOX':
            return app.ForceField(ff_file_name, 'charmm36/water.xml')
        elif water_model_name == 'SPCEBOX':
            return app.ForceField(ff_file_name, 'charmm36/spce.xml')
        elif water_model_name == 'TIP4PEWBOX':
            return app.ForceField(ff_file_name, 'charmm36/tip4pew.xml')
        elif water_model_name == 'TIP5PBOX':
            return app.ForceField(ff_file_name, 'charmm36/tip5p.xml')
        elif water_model_name is None:
            return app.ForceField(ff_file_name)
        else:
            raise ValueError('%s Water model not found for force field %s' % (water_model_name, ff_name))


"""
Simulation Functions
"""


def run_prep(pdb_source, pdb_target, topology_target, buffer_size=10.0, force_field_name='ff99SBildn',
             water_model_name='TIP3PBOX', cation='Na+', anion='Cl-'):
    """

    :param pdb_source:
    :param pdb_target:
    :param topology_target:
    :param buffer_size:
    :param force_field_name:
    :param water_model_name:
    :param cation:
    :param anion:
    :return:
    """

    wm_map = {
        'TIP3PBOX': 'tip3p',
        'SPCEBOX': 'spce',
        'TIP4PEWBOX': 'tip4pew',
        'TIP5PBOX': 'tip5p',
    }
    pdb = app.PDBFile(pdb_source)
    mdl = app.Modeller(pdb.topology, pdb.positions)
    forcefield = get_force_field(force_field_name, water_model_name)

    mdl.addSolvent(forcefield, model=wm_map[water_model_name], padding=buffer_size / 10,
                   positiveIon=cation, negativeIon=anion)

    system = forcefield.createSystem(
        mdl.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * (unit.nano * unit.meter),
        constraints=app.HBonds,
        rigidWater=True,
    )

    # target 확장자명이 gz인 경우 gzip 압축 실행
    if 'gz' == topology_target[-2:]:
        with gzip.open(topology_target, 'wt', encoding='utf-8') as f:
            f.write(mm.XmlSerializer.serialize(system))
    else:
        with open(topology_target, 'w', encoding='utf-8') as f:
            f.write(mm.XmlSerializer.serialize(system))

    temp_dir = tempfile.mkdtemp()
    with open(os.path.join(temp_dir, 'model_solv.pdb'), 'w', encoding='utf-8') as f:
        app.PDBFile.writeFile(mdl.topology, mdl.positions, f)

    t = mdt.load(os.path.join(temp_dir, 'model_solv.pdb'))
    t.image_molecules()
    t.center_coordinates()

    # target 확장자명이 gz인 경우 gzip 압축 실행
    if 'gz' == pdb_target[-2:]:
        with gzip.open(pdb_target, 'wt', encoding='utf-8') as f:
            app.PDBFile.writeFile(mdl.topology, t.xyz[0] * 10.0, f)
    else:
        with open(pdb_target, 'w', encoding='utf-8') as f:
            app.PDBFile.writeFile(mdl.topology, t.xyz[0] * 10.0, f)

    shutil.rmtree(temp_dir)


def run_min(topology_source, pdb_source, maxcyc, state_file, restraint_weight=None, reference=None):
    """
    :param topology_source:
    :param pdb_source:
    :param maxcyc: Max minimization steps
    :param state_file: npz file path to save State object of last frame
    :param restraint_weight:
    :param reference: npz file path for starting State of Simulation
    :return:
    """
    if 'gz' == topology_source[-2:]:
        with gzip.open(topology_source, 'rt', encoding='utf-8') as f:
            system = mm.XmlSerializer.deserialize(f.read())
    else:
        with open(topology_source, 'r', encoding='utf-8') as f:
            system = mm.XmlSerializer.deserialize(f.read())

    if 'gz' == pdb_source[-2:]:
        with gzip.open(pdb_source, 'rt', encoding='utf-8') as f:
            ps = app.pdbfile.PdbStructure(f)
            pdb = app.PDBFile(ps)
    else:
        pdb = app.PDBFile(pdb_source)

    if restraint_weight and isinstance(restraint_weight, float):
        system.addForce(get_restrainted_force(restraint_weight, pdb))

    integrator = mm.LangevinIntegrator(300 * unit.kelvin, 1.0 / (unit.pico * unit.second),
                                       0.002 * (unit.pico * unit.second))
    integrator.setConstraintTolerance(0.00001)

    platform = get_platform()
    simulation = app.Simulation(pdb.topology, system, integrator, platform)

    if reference:
        _ref = np.load(reference, allow_pickle=True)
        simulation.context.setPositions(_ref['Positions'])
        simulation.context.setPeriodicBoxVectors(*_ref['PeriodicBoxVectors'])
    else:
        simulation.context.setPositions(pdb.positions)

    simulation.minimizeEnergy(maxIterations=maxcyc)

    st = simulation.context.getState(getPositions=True, getEnergy=True)

    _pos = st.getPositions()
    bv = st.getPeriodicBoxVectors()

    np.savez(state_file, Positions=_pos, PeriodicBoxVectors=bv)

    return pdb.topology, st


def run_eq(topology_source, pdb_source, nstlim, out_file, state_file, reference,
           tempi=0.0, temp0=300.0, dt=0.002, ntpr=2500, ntb=1, restraint_weight=None):
    """
    :param topology_source:
    :param pdb_source:
    :param nstlim: Simulation steps
    :param out_file: target file to save StateDataReporter output
    :param state_file: npz file path to save State object of last frame
    :param reference: npz file path for starting State of Simulation
    :param tempi: Initial temperature
    :param temp0: Reference temperature
    :param dt: Time step
    :param ntpr: StateDataReporter logging interval
    :param ntb: 1 for NVT, 2 for NPT
    :param restraint_weight:
    :return:
    """
    if 'gz' == topology_source[-2:]:
        with gzip.open(topology_source, 'rt', encoding='utf-8') as f:
            system = mm.XmlSerializer.deserialize(f.read())
    else:
        with open(topology_source, 'r', encoding='utf-8') as f:
            system = mm.XmlSerializer.deserialize(f.read())

    if 'gz' == pdb_source[-2:]:
        with gzip.open(pdb_source, 'rt', encoding='utf-8') as f:
            ps = app.pdbfile.PdbStructure(f)
            pdb = app.PDBFile(ps)
    else:
        pdb = app.PDBFile(pdb_source)

    if restraint_weight and isinstance(restraint_weight, float):
        system.addForce(get_restrainted_force(restraint_weight, pdb))

    integrator = mm.LangevinIntegrator(temp0 * unit.kelvin, 1.0 / (unit.pico * unit.second),
                                       dt * (unit.pico * unit.second))
    integrator.setConstraintTolerance(0.00001)

    if ntb == 2:
        system.addForce(mm.MonteCarloBarostat(1 * unit.atmospheres, temp0, 25))

    platform = get_platform()
    simulation = app.Simulation(pdb.topology, system, integrator, platform)
    _ref = np.load(reference, allow_pickle=True)
    simulation.context.setPositions(_ref['Positions'])
    simulation.context.setPeriodicBoxVectors(*_ref['PeriodicBoxVectors'])
    if 'Velocities' in _ref:
        simulation.context.setVelocities(_ref['Velocities'])
    else:
        simulation.context.setVelocitiesToTemperature(tempi * unit.kelvin)

    # simulation.reporters.append(app.DCDReporter(traj_file, ntwx))
    simulation.reporters.append(app.StateDataReporter(out_file, ntpr, step=True,
                                                      time=True, potentialEnergy=True, kineticEnergy=True,
                                                      totalEnergy=True,
                                                      temperature=True, volume=True, density=True,
                                                      progress=True,
                                                      remainingTime=True, speed=True, totalSteps=nstlim,
                                                      separator=' '))
    simulation.step(nstlim)

    st = simulation.context.getState(getPositions=True, getVelocities=True)
    pos = st.getPositions()
    vel = st.getVelocities()
    bv = st.getPeriodicBoxVectors()
    np.savez(state_file, Positions=pos, Velocities=vel, PeriodicBoxVectors=bv)

    return pdb.topology, st


def run_md(topology_source, pdb_source, nstlim, out_file, traj_file, state_file, reference,
           temp0=300.0, dt=0.002, ntwx=2500, ntpr=2500, ntb=1, pres0=1.0):
    """
    :param topology_source:
    :param pdb_source:
    :param nstlim: Simulation steps
    :param out_file: target file path to save StateDataReporter output
    :param traj_file: trajectory file path
    :param state_file: npz file path to save State object of last frame
    :param reference: npz file path for starting State of Simulation
    :param temp0: Reference temperature
    :param dt: Time step
    :param ntwx: trajectory saving interval of each frame
    :param ntpr: StateDataReporter logging interval
    :param ntb: 1 for NVT, 2 for NPT
    :param pres0: reference pressure to be kept during NPT Simulation
    :return:
    """
    if 'gz' == topology_source[-2:]:
        with gzip.open(topology_source, 'rt', encoding='utf-8') as f:
            system = mm.XmlSerializer.deserialize(f.read())
    else:
        with open(topology_source, 'r', encoding='utf-8') as f:
            system = mm.XmlSerializer.deserialize(f.read())

    if 'gz' == pdb_source[-2:]:
        with gzip.open(pdb_source, 'rt', encoding='utf-8') as f:
            ps = app.pdbfile.PdbStructure(f)
            pdb = app.PDBFile(ps)
    else:
        pdb = app.PDBFile(pdb_source)

    integrator = mm.LangevinIntegrator(temp0 * unit.kelvin, 1.0 / (unit.pico * unit.second),
                                       dt * (unit.pico * unit.second))
    integrator.setConstraintTolerance(0.00001)

    if ntb == 2:
        system.addForce(mm.MonteCarloBarostat(pres0 * unit.atmospheres, temp0, 25))

    platform = get_platform()
    simulation = app.Simulation(pdb.topology, system, integrator, platform)
    _ref = np.load(reference, allow_pickle=True)
    simulation.context.setPositions(_ref['Positions'])
    simulation.context.setPeriodicBoxVectors(*_ref['PeriodicBoxVectors'])
    simulation.context.setVelocities(_ref['Velocities'])
    simulation.reporters.append(app.DCDReporter(traj_file, ntwx))
    simulation.reporters.append(app.StateDataReporter(out_file, ntpr, step=True,
                                                      time=True, potentialEnergy=True, kineticEnergy=True,
                                                      totalEnergy=True,
                                                      temperature=True, volume=True, density=True,
                                                      progress=True,
                                                      remainingTime=True, speed=True, totalSteps=nstlim,
                                                      separator=' '))
    simulation.step(nstlim)
    st = simulation.context.getState(getPositions=True, getVelocities=True)

    pos = st.getPositions()
    vel = st.getVelocities()
    bv = st.getPeriodicBoxVectors()
    np.savez(state_file, Positions=pos, Velocities=vel, PeriodicBoxVectors=bv)

    return pdb.topology, st


"""
WebMD Integration
"""


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError:
        pass


def webmd_prep(work_dir, buffer_size=10.0, ff_name='ff99SBildn', wm_name='TIP3PBOX', cation='Na+', anion='Cl-'):
    base_dir = os.path.join(work_dir, 'prep')
    mkdir(base_dir)

    # write input file
    # with open(os.path.join(base_dir, 'input.json'), 'w') as f:
    #     json.dump({
    #         # 'work_dir': work_dir,
    #         'solvent_model': wm_name,
    #         'buffer_size': buffer_size,
    #         'cation': cation,
    #         'anion': anion,
    #     }, f, indent=2)

    run_prep(pdb_source=os.path.join(work_dir, 'model.pdb'),
             pdb_target=os.path.join(base_dir, 'model_solv.pdb'),
             topology_target=os.path.join(base_dir, 'model.xml'),
             buffer_size=buffer_size,
             force_field_name=ff_name,
             water_model_name=wm_name,
             cation=cation, anion=anion)


def webmd_min(work_dir, maxcyc1=1000, maxcyc2=2500):
    base_dir = os.path.join(work_dir, 'min')
    mkdir(base_dir)

    # write input file
    with open(os.path.join(base_dir, 'min.in'), 'w') as f:
        json.dump({
            'min1': {
                'maxcyc': maxcyc1,
                'restraint_wt': 10.0,
            },
            'min2': {
                'maxcyc': maxcyc2,
            }
        }, f, indent=2)

    # min1
    t1, st1 = run_min(pdb_source=os.path.join(work_dir, 'prep/model_solv.pdb.gz'),
                      topology_source=os.path.join(work_dir, 'prep/model.xml'),
                      maxcyc=maxcyc1,
                      state_file=os.path.join(base_dir, 'min1.npz'),
                      restraint_weight=10.0)

    temp_dir = tempfile.mkdtemp()
    with open(os.path.join(temp_dir, 'min1.pdb'), 'w', encoding='utf-8') as f:
        app.PDBFile.writeFile(t1, st1.getPositions(), f)

    t = mdt.load(os.path.join(temp_dir, 'min1.pdb'))
    t.image_molecules()
    t.center_coordinates()
    with gzip.open(os.path.join(base_dir, 'min1.pdb.gz'), 'wt') as f:
        app.PDBFile.writeFile(t1, t.xyz[0] * 10.0, f)

    # min2
    t2, st2 = run_min(pdb_source=os.path.join(work_dir, 'prep/model_solv.pdb.gz'),
                      topology_source=os.path.join(work_dir, 'prep/model.xml'),
                      maxcyc=maxcyc2,
                      state_file=os.path.join(base_dir, 'min2.npz'),
                      reference=os.path.join(base_dir, 'min1.npz'))

    with open(os.path.join(temp_dir, 'min2.pdb'), 'w', encoding='utf-8') as f:
        app.PDBFile.writeFile(t2, st2.getPositions(), f)

    t = mdt.load(os.path.join(temp_dir, 'min2.pdb'))
    t.image_molecules()
    t.center_coordinates()
    with gzip.open(os.path.join(base_dir, 'min2.pdb.gz'), 'wt') as f:
        app.PDBFile.writeFile(t2, t.xyz[0] * 10.0, f)

    shutil.rmtree(temp_dir)


def webmd_eq(work_dir, nstlim1=10000, nstlim2=100000, tempi=0.0, temp0=300.0):
    base_dir = os.path.join(work_dir, 'eq')
    mkdir(base_dir)

    # write input file
    with open(os.path.join(base_dir, 'eq1.in'), 'w') as f:
        json.dump({
            'eq1': {
                'dt': 0.002,
                'nstlim': nstlim1,
                'tempi': tempi,
                'temp0': temp0,
                'ntb': 1,
                'ntpr': 2500,
                'restraint_wt': 10.0,
            },
            'eq2': {
                'dt': 0.002,
                'nstlim': nstlim2,
                'temp0': temp0,
                'ntb': 2,
                'ntpr': 2500,
            }
        }, f, indent=2)

    # eq1
    t1, st1 = run_eq(pdb_source=os.path.join(work_dir, 'prep/model_solv.pdb.gz'),
                     topology_source=os.path.join(work_dir, 'prep/model.xml'),
                     nstlim=nstlim1,
                     out_file=os.path.join(base_dir, 'eq1.out'),
                     state_file=os.path.join(base_dir, 'eq1.npz'),
                     reference=os.path.join(work_dir, 'min/min2.npz'),
                     tempi=tempi,
                     temp0=temp0,
                     restraint_weight=10.0)

    temp_dir = tempfile.mkdtemp()
    with open(os.path.join(temp_dir, 'eq1.pdb'), 'w', encoding='utf-8') as f:
        app.PDBFile.writeFile(t1, st1.getPositions(), f)

    t = mdt.load(os.path.join(temp_dir, 'eq1.pdb'))
    t.image_molecules()
    t.center_coordinates()
    with gzip.open(os.path.join(base_dir, 'eq1.pdb.gz'), 'wt') as f:
        app.PDBFile.writeFile(t1, t.xyz[0] * 10.0, f)

    # eq2
    t2, st2 = run_eq(pdb_source=os.path.join(work_dir, 'prep/model_solv.pdb.gz'),
                     topology_source=os.path.join(work_dir, 'prep/model.xml'),
                     nstlim=nstlim2,
                     out_file=os.path.join(base_dir, 'eq2.out'),
                     state_file=os.path.join(base_dir, 'eq2.npz'),
                     reference=os.path.join(base_dir, 'eq1.npz'),
                     temp0=temp0,
                     ntb=2)

    with open(os.path.join(temp_dir, 'eq2.pdb'), 'w', encoding='utf-8') as f:
        app.PDBFile.writeFile(t2, st2.getPositions(), f)

    t = mdt.load(os.path.join(temp_dir, 'eq2.pdb'))
    t.image_molecules()
    t.center_coordinates()
    with gzip.open(os.path.join(base_dir, 'eq2.pdb.gz'), 'wt') as f:
        app.PDBFile.writeFile(t2, t.xyz[0] * 10.0, f)

    shutil.rmtree(temp_dir)


def webmd_md(work_dir, nstlim=100000, temp0=300.0, ntb=1, pres0=1.0):
    def md_serials():
        md_root_dir = os.path.join(work_dir, 'md')
        if os.path.exists(md_root_dir):
            ret = []
            for serial in next(os.walk(md_root_dir))[1]:
                md_npz = os.path.join(md_root_dir, '%s/md.npz' % serial)
                if os.path.exists(md_npz):
                    ret.append(int(serial))
            return sorted(ret)
        else:
            return []

    mdserial = len(md_serials()) + 1

    base_dir = os.path.join(work_dir, 'md/%d' % mdserial)
    mkdir(base_dir)

    # write input file
    with open(os.path.join(base_dir, 'md.in'), 'w') as f:
        json.dump({
            'dt': 0.002,
            'nstlim': nstlim,
            'temp0': temp0,
            'ntb': ntb,
            'pres0': pres0,
            'ntwx': 2500,
            'ntpr': 2500,
        }, f, indent=2)

    reference = os.path.join(work_dir, 'eq/eq2.npz')
    if mdserial > 1:
        reference = os.path.join(work_dir, 'md/%d/md.npz' % (mdserial-1))

    temp_dir = tempfile.mkdtemp()

    run_md(pdb_source=os.path.join(work_dir, 'prep/model_solv.pdb.gz'),
           topology_source=os.path.join(work_dir, 'prep/model.xml'),
           nstlim=nstlim,
           out_file=os.path.join(base_dir, 'md.out'),
           traj_file=os.path.join(temp_dir, 'md.dcd'),
           state_file=os.path.join(base_dir, 'md.npz'),
           reference=reference,
           temp0=temp0,
           ntb=ntb, pres0=pres0)

    t = mdt.load(os.path.join(temp_dir, 'md.dcd'), top=os.path.join(work_dir, 'prep/model_solv.pdb.gz'))
    t.image_molecules()
    t.center_coordinates()

    with gzip.open(os.path.join(base_dir, 'md.pdb.gz'), 'wt') as f:
        app.PDBFile.writeFile(t.topology.to_openmm(), t.xyz[0] * 10.0, f)

    shutil.move(os.path.join(temp_dir, 'md.dcd'), os.path.join(base_dir, 'md.dcd'))
    shutil.rmtree(temp_dir)


"""
Analysis functions
"""


def get_masses(t):
    """
    get mass of each atom

    :param t: trajectory
    :return:
    """
    masses = []
    for a in t.topology.atoms:
        masses.append(a.element.mass)
    return np.array(masses)


def calculate_rmsd(reference_file, topology_file, out_file, trajin=None, **kwargs):
    r = mdt.load(reference_file, top=topology_file)
    sel = r.topology.select('protein and name CA')

    with open(out_file, 'w') as f:
        f.write('# rmsd\n')
        f.write('%.4f\n' % (mdt.rmsd(r, r, atom_indices=sel, precentered=True) * 10.0))

    for traj in trajin:
        t = mdt.load(traj, top=topology_file)
        t.image_molecules(inplace=True)
        t.center_coordinates(mass_weighted=False)
        t.superpose(r, frame=0, atom_indices=sel)
        with open(out_file, 'a') as f:
            for val in (mdt.rmsd(t, r, atom_indices=sel, precentered=True) * 10.0):
                f.write('%.4f\n' % val)


def calculate_rmsf(reference_file, topology_file, out_file, trajin=None, **kwargs):
    r = mdt.load(reference_file, top=topology_file)
    sel = r.topology.select('protein and name CA')

    coords = []
    for traj in trajin:
        t = mdt.load(traj, top=topology_file)
        t.image_molecules(inplace=True)
        t.center_coordinates(mass_weighted=False)
        t.superpose(r, frame=0, atom_indices=sel)

        tsel = t.xyz[:, sel, :]
        coords.append(tsel)

    xyz = np.concatenate(coords, axis=0)
    refxyz = np.mean(r.xyz[:, sel, :], axis=0)

    rmsf = np.sqrt(3.0 * np.mean((xyz - refxyz) ** 2, axis=(0, 2))) * 10.0  # nm to Angstrom

    atoms = list(r.topology.subset(sel).to_openmm().atoms())

    with open(out_file, 'w') as f:
        print('# atom RMSF', file=f)
        for i in range(len(sel)):
            resnum = atoms[i].residue.id
            print('{:6s} {:.4f}'.format(resnum, rmsf[i]), file=f)


def calculate_radgyr(reference_file, topology_file, out_file, mask='protein', trajin=None):
    def compute_rg(_t, _sel, _mass_weighted):
        num_atoms = _sel.size
        if _mass_weighted:
            masses = get_masses(_t)[_sel]
        else:
            masses = np.ones(num_atoms)

        xyz = _t.xyz[:, _sel, :]
        weights = masses / masses.sum()

        mu = xyz.mean(1)
        centered = (xyz.transpose((1, 0, 2)) - mu).transpose((1, 0, 2))
        squared_dists = (centered ** 2).sum(2)
        return (squared_dists * weights).sum(1) ** 0.5 * 10.0  # nm to Angstrom

    r = mdt.load(reference_file, top=topology_file)
    sel = r.topology.select(mask)

    with open(out_file, 'w') as f:
        f.write('# radgyr')
        f.write('%.4f\n' % compute_rg(r, sel, False))

    for traj in trajin:
        trajs = mdt.load(traj, top=topology_file)
        trajs.image_molecules(inplace=True)
        trajs.center_coordinates(mass_weighted=False)
        trajs.superpose(r, frame=0, atom_indices=sel)

        for t in trajs:
            with open(out_file, 'a') as f:
                f.write('%.4f\n' % compute_rg(t, sel, False))


def calculate_sasa(reference_file, topology_file, out_file, mask='protein', trajin=None):
    r = mdt.load(reference_file, top=topology_file)
    sel = r.topology.select(mask)

    with open(out_file, 'w') as f:
        f.write('# sasa')
        f.write('%.4f\n' % mdt.shrake_rupley(r).sum(axis=1))

    for traj in trajin:
        trajs = mdt.load(traj, top=topology_file)
        trajs.image_molecules(inplace=True)
        trajs.center_coordinates(mass_weighted=False)
        trajs.superpose(r, frame=0, atom_indices=sel)

        for t in trajs:
            with open(out_file, 'a') as f:
                f.write('%.4f\n' % mdt.shrake_rupley(t).sum(axis=1))


def calculate_eu(reference_file, topology_file, out_file, mask='protein', forcefield='ff99SBildn', trajin=None):
    def compute_eu(_t, _sel, _system, _topo):
        pos = _t.xyz[:, _sel, ][0]
        integrator = mm.LangevinIntegrator(300, 1.0 / (unit.pico * unit.second), 0.002 * (unit.pico * unit.second))
        simulation = app.Simulation(_topo, _system, integrator)
        simulation.context.setPositions(pos)
        st = simulation.context.getState(getEnergy=True)
        return st.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)

    r = mdt.load(reference_file, top=topology_file)
    sel = r.topology.select(mask)

    topo = r.topology.subset(sel).to_openmm()
    system = get_force_field(forcefield, water_model_name=None).createSystem(topo)

    with open(out_file, 'w') as f:
        f.write('#Time\tPotential_energy\n')
        f.write('0\t%f\n' % compute_eu(r, sel, system, topo))

    time = 5
    for traj in trajin:
        trajs = mdt.load(traj, top=topology_file)
        trajs.image_molecules(inplace=True)
        trajs.center_coordinates()
        for t in trajs:
            with open(out_file, 'a') as f:
                f.write('%d\t%f\n' % (time, compute_eu(t, sel, system, topo)))
            time += 5


def run_rism3dx(rism_input_file):
    base_dir = os.path.dirname(os.path.abspath(rism_input_file))
    out_f = open(os.path.join(base_dir, 'rism3dx.out.log'), 'w')
    err_f = open(os.path.join(base_dir, 'rism3dx.err.log'), 'w')
    subprocess.call([
        os.environ.get('RISM3DX_EXE', '/opt/nbcc/bin/rism3d-x'),
        os.path.abspath(rism_input_file),
        os.environ.get('CUDA_VISIBLE_DEVICES', '0'),
    ], stdout=out_f, stderr=err_f)
    out_f.close()
    err_f.close()


def webmd_calculate_gsolv(reference_file, topology_file, out_file, mask='protein', stride=5, trajin=None):
    def compute_gsolv(_topo, _pos, i):
        generate_rism_input(os.path.join(base_dir, 'frame_%d.rsm' % i), _topo, _pos, 'm', 128.0, 128)
        run_rism3dx(os.path.join(base_dir, 'frame_%d.rsm' % i))
        shutil.move(os.path.join(base_dir, 'rism3dx.out.log'), os.path.join(base_dir, 'logs/frame_%d.out.log' % i))
        shutil.move(os.path.join(base_dir, 'rism3dx.err.log'), os.path.join(base_dir, 'logs/frame_%d.error.log' % i))

        with open(os.path.join(base_dir, 'frame_%d.xmu' % i), 'r') as g:
            for line in g:
                key, value = line.split()
                if key == 'solvation_free_energy':
                    return float(value)
        return None

    base_dir = os.path.dirname(os.path.abspath(out_file))
    mkdir(os.path.join(base_dir, 'logs'))

    r = mdt.load(reference_file, top=topology_file)
    sel = r.topology.select(mask)
    topo = r.topology.subset(sel)

    with open(os.path.abspath(out_file), 'w') as f:
        f.write('#frame\tsolvation_free_energy\n')

    ts = []
    for traj in trajin:
        t = mdt.load(traj, top=topology_file)
        t.image_molecules(inplace=True)
        t.topology = t.topology.subset(sel)
        t.xyz = t.xyz[:, sel, :]
        ts.append(t)

    trajs = ts[0].join(ts[1:])

    for nframe, t in enumerate(trajs):
        if nframe % stride == (stride-1):  # 5 th, 10 th, 15 th ... frame
            t.center_coordinates()
            gsolv = compute_gsolv(topo, t.xyz[0], nframe)
            with open(os.path.abspath(out_file), 'a') as f:
                f.write('%8d\t%.4f\n' % (nframe, gsolv))


def prowave_calcuate_gsolv(work_dir, mode, box_size=128.0, grid_size=128, forcefield='ff99SBildn'):
    pdb_file = os.path.join(work_dir, 'model.pdb')
    t = mdt.load(pdb_file)
    sel = t.topology.select('protein')
    topo = t.topology.subset(sel)
    pos = t.xyz[0]
    base_dir = os.path.join(work_dir, 'analyses/1')
    mkdir(base_dir)
    cwd = os.getcwd()
    os.chdir(base_dir)
    generate_rism_input(os.path.join(base_dir, 'frame_0.rsm'), topo, pos, mode, box_size, grid_size,
                        forcefield=forcefield)
    run_rism3dx(os.path.join(base_dir, 'frame_0.rsm'))
    post_process(base_dir, mode, 0)
    os.chdir(cwd)


def generate_rism_input(out_file, topo: mdt.Topology, pos, mode: str, box_size: float, grid_size: int,
                        water_function='tip3p_combined_300K', forcefield='ff99SBildn'):
    ff = get_force_field(ff_name=forcefield)
    system = ff.createSystem(topo.to_openmm())
    header = """20130208C
{0}
KH
/opt/nbcc/common/3D-RISM/{1}.xsv
1.0e-6  10000
{2:5.1f}   {2:5.1f}   {2:5.1f}
{3}   {3}   {3}
""".format(mode, water_function, box_size, grid_size)

    for idx in range(system.getNumForces()):
        nbforce = system.getForce(idx)
        if isinstance(nbforce, mm.openmm.NonbondedForce):
            break

    coord = pos * 10.0

    # charge / sigma / epsilon
    natom = pos.shape[0]
    joule_per_mole = unit.joule / unit.mole
    charge = np.empty(natom, dtype=np.double)
    sigma = np.empty(natom, dtype=np.double)
    epsilon = np.empty(natom, dtype=np.double)
    for idx in range(natom):
        atmind = topo.atom(idx).index
        p = nbforce.getParticleParameters(atmind)
        charge[idx] = p[0].value_in_unit(unit.elementary_charge)
        sigma[idx] = p[1].value_in_unit(unit.angstrom)
        epsilon[idx] = p[2].value_in_unit(joule_per_mole)

    with open(out_file, 'w') as f:
        f.write(header)
        f.write('{:5d}\n'.format(natom))
        for i in range(natom):
            atmind = topo.atom(i).index
            x, y, z = coord[atmind]
            f.write(' {:16.5f}{:16.7f}{:16.7f}{:16.7f}{:16.7f}{:16.7f}\n'.format(charge[i], sigma[i], epsilon[i],
                                                                                 x, y, z))


def process_xmu(base_dir, result, i=0):
    """

    :param base_dir:
    :param result: result.out file stream
    :param i: frame index number (Optional)
    :return:
    """
    with open(os.path.join(base_dir, 'frame_%d.xmu' % i), 'r') as f:
        for line in f:
            key, value = line.split()
            if key == 'solvation_free_energy':
                print('%s\t%s' % (key, value), file=result)
                break


def process_thm(base_dir, result, i=0):
    """

    :param base_dir:
    :param result: result.out file stream
    :param i: frame index number (Optional)
    :return:
    """
    with open(os.path.join(base_dir, 'frame_%d.thm' % i), 'r') as f:
        for line in f:
            key, value = line.split()
            if key in ('solvation_energy', 'solvation_entropy'):
                print('%s\t%s' % (key, value), file=result)
                continue


def process_xmu_res(base_dir, result, resmap):
    """

    :param base_dir:
    :param result: result.out file stream
    :param resmap: (atom_index, residue name) dictionary
    :return:
    """
    xmua_atm_file = os.path.join(base_dir, 'xmua_atm.dat')
    with open(xmua_atm_file, 'r') as f:
        data = f.readlines()[2:]

    resultmap = dict()
    for d in data:
        atom_ind, sfe, _, _ = d.split()

        try:
            residue = resmap[atom_ind]
        except KeyError:
            continue

        if residue not in resultmap:
            resultmap[residue] = 0.0
        resultmap[residue] += float(sfe)

    for key, value in resultmap.items():
        result.write('%s\t%f\n' % (key, value))


def process_thm_res(base_dir, result, resmap):
    """

    :param base_dir:
    :param result: result.out file stream
    :param resmap: (atom_index, residue name) dictionary
    :return:
    """
    xmua_atm = os.path.join(base_dir, 'xmua_atm.dat')
    enea_atm = os.path.join(base_dir, 'enea_atm.dat')
    enta_atm = os.path.join(base_dir, 'enta_atm.dat')

    xmua_atm_f = open(xmua_atm, 'r')
    enea_atm_f = open(enea_atm, 'r')
    enta_atm_f = open(enta_atm, 'r')

    xmu_data = xmua_atm_f.readlines()[2:]
    ene_data = enea_atm_f.readlines()[2:]
    ent_data = enta_atm_f.readlines()[2:]

    resultmap = dict()
    for xmu_d, ene_d, ent_d in zip(xmu_data, ene_data, ent_data):
        xmu = xmu_d.split()
        ene = ene_d.split()
        ent = ent_d.split()

        try:
            residue = resmap[xmu[0]]
        except KeyError:
            continue

        if residue not in resultmap:
            resultmap[residue] = [0.0, 0.0, 0.0]
        resultmap[residue][0] += float(xmu[1])
        resultmap[residue][1] += float(ene[1])
        resultmap[residue][2] += float(ent[1])

    for key, values in resultmap.items():
        result.write('%s\t%f\t%f\t%f\n' % (key, values[0], values[1], values[2]))

    xmua_atm_f.close()
    enea_atm_f.close()
    enta_atm_f.close()


def post_process(base_dir, mode, i=0):
    res_map = dict()
    with open(os.path.join(base_dir, '../../model.pdb'), 'r') as pdb:
        for line in pdb:
            if 'ATOM' in line[0:6]:
                atom_ind, chain, resnum = line[6:11].strip(), line[21:22].strip(), line[22:26].strip()
                res_map[atom_ind] = '%s %s' % (chain, resnum)

    with open(os.path.join(base_dir, 'result.out'), 'w') as result:
        if mode == 'm':
            process_xmu(base_dir, result, i)
        elif mode == 't':
            process_xmu(base_dir, result, i)
            process_thm(base_dir, result, i)
        elif mode == 'a':
            process_xmu(base_dir, result, i)
            process_xmu_res(base_dir, result, res_map)
        elif mode == 'x':
            process_xmu(base_dir, result, i)
            process_thm(base_dir, result, i)
            process_thm_res(base_dir, result, res_map)

    subprocess.check_call(['/bin/cat', os.path.join(base_dir, 'result.out')])


"""
Job scheduling method
"""


def submit_batch(base_dir, func, dependency=None, partition=None, **kwargs):
    def parse_subcmd(_func, args_map):
        if _func == 'prep':
            ret = [args_map['pdb_source']]
            for param in ('pdb_target', 'topology_target', 'buffer_size', 'forcefield', 'watermodel', 'cation', 'anion'):
                if param in args_map:
                    ret += ['--%s' % param, str(args_map[param])]
            return ret

        elif _func == 'min':
            ret = [args_map['maxcyc']]
            for param in ('topology_source', 'pdb_source', 'state_file', 'restraint_weight', 'reference'):
                if param in args_map:
                    ret += ['--%s' % param, str(args_map[param])]
            return ret

        elif _func == 'eq':
            ret = [args_map['reference']]
            for param in ('nstlim', 'topology_source', 'pdb_source', 'state_file', 'out_file', 'tempi', 'temp0',
                          'dt', 'ntpr', 'ntb', 'restraint_weight'):
                if param in args_map:
                    ret += ['--%s' % param, str(args_map[param])]
            return ret

        elif _func == 'md':
            ret = [args_map['reference']]
            for param in ('nstlim', 'topology_source', 'pdb_source', 'state_file', 'out_file', 'traj_file', 'temp0',
                          'dt', 'ntpr', 'ntwx', 'ntb', 'pres0'):
                if param in args_map:
                    ret += ['--%s' % param, str(args_map[param])]
            return ret

        elif _func == 'webmd_prep':
            ret = [args_map['work_dir']]
            for param in ('buffer_size', 'forcefield', 'watermodel', 'cation', 'anion'):
                if param in args_map:
                    ret += ['--%s' % param, str(args_map[param])]
            return ret

        elif _func == 'webmd_min':
            ret = [args_map['work_dir']]
            for param in ('maxcyc1', 'maxcyc2'):
                if param in args_map:
                    ret += ['--%s' % param, str(args_map[param])]
            return ret

        elif _func == 'webmd_eq':
            ret = [args_map['work_dir']]
            for param in ('nstlim1', 'nstlim2', 'tempi', 'temp0'):
                if param in args_map:
                    ret += ['--%s' % param, str(args_map[param])]
            return ret

        elif _func == 'webmd_md':
            ret = [args_map['work_dir']]
            for param in ('nstlim', 'temp0', 'ntb', 'pres0'):
                if param in args_map:
                    ret += ['--%s' % param, str(args_map[param])]
            return ret

        elif _func == 'anal':
            ret = [args_map['method'], args_map['reference_file'], args_map['topology_file'], args_map['out_file']]
            for param in ('mask', 'forcefield'):
                if param in args_map:
                    ret += ['--%s' % param, str(args_map[param])]

            if 'trajin' in args_map:
                ret += ['--trajin'] + args_map['trajin']

            return ret
        elif _func == 'prowave':
            ret = [args_map['work_dir'], args_map['mode']]
            for param in ('box_size', 'grid_size'):
                if param in args_map:
                    ret += ['--%s' % param, str(args_map[param])]
            return ret
        elif _func == 'rism3dx':
            return [args_map['rism_input_file']]
        return []

    mkdir(base_dir)
    stdout = os.path.join(base_dir, 'stdout')
    stderr = os.path.join(base_dir, 'stderr')

    sbatch_cmd = os.path.join(os.environ.get('SLURM_HOME', '/usr/local'), 'bin/sbatch')

    cmd = [
        sbatch_cmd,
        '--output', stdout,
        '--error', stderr,
        '--nodes', '1',
        '--time', '72:00:00',
        '--job-name', func,
        '--ntasks', '1',
        '--gres', 'gpu:1'
    ]

    if partition:
        cmd += ['--partition', partition]
    else:
        cmd += ['--partition', 'prowave']

    if dependency:
        cmd += ['--dependency', dependency]

    cmd += [os.path.abspath(__file__), func] + parse_subcmd(func, kwargs)

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

    parsers = dict()
    parsers['prep'] = subparsers.add_parser('prep')
    parsers['prep'].add_argument('pdb_source')
    parsers['prep'].add_argument('--pdb_target', default='model_solv.pdb.gz')
    parsers['prep'].add_argument('--topology_target', default='model.xml.gz')
    parsers['prep'].add_argument('--buffer_size', default=10.0, type=float)
    parsers['prep'].add_argument('--forcefield', default='ff99SBildn')
    parsers['prep'].add_argument('--watermodel', default='TIP3PBOX')
    parsers['prep'].add_argument('--cation', default='Na+')
    parsers['prep'].add_argument('--anion', default='Cl-')

    parsers['min'] = subparsers.add_parser('min')
    parsers['min'].add_argument('maxcyc', type=int)
    parsers['min'].add_argument('--topology_source', default='model.xml.gz')
    parsers['min'].add_argument('--pdb_source', default='model_solv.pdb.gz')
    parsers['min'].add_argument('--state_file', default='min.npz')
    parsers['min'].add_argument('-w', '--restraint_weight', type=float)
    parsers['min'].add_argument('-r', '--reference')

    parsers['eq'] = subparsers.add_parser('eq')
    parsers['eq'].add_argument('reference')
    parsers['eq'].add_argument('-n', '--nstlim', type=int, default=10000)
    parsers['eq'].add_argument('--topology_source', default='model.xml.gz')
    parsers['eq'].add_argument('--pdb_source', default='model_solv.pdb.gz')
    parsers['eq'].add_argument('--state_file', default='eq.npz')
    parsers['eq'].add_argument('--out_file', default='eq.out')
    parsers['eq'].add_argument('--tempi', type=float, default=0.0)
    parsers['eq'].add_argument('--temp0', type=float, default=300.0)
    parsers['eq'].add_argument('--dt', type=float, default=0.002)
    parsers['eq'].add_argument('--ntpr', type=int, default=2500)
    parsers['eq'].add_argument('--ntb', type=int, default=1)
    parsers['eq'].add_argument('-w', '--restraint_weight', type=float)

    parsers['md'] = subparsers.add_parser('md')
    parsers['md'].add_argument('reference')
    parsers['md'].add_argument('--topology_source', default='model.xml.gz')
    parsers['md'].add_argument('--pdb_source', default='model_solv.pdb.gz')
    parsers['md'].add_argument('-n', '--nstlim', type=int, default=500000)
    parsers['md'].add_argument('--out_file', default='md.out')
    parsers['md'].add_argument('--traj_file', default='md.dcd')
    parsers['md'].add_argument('--state_file', default='md.npz')
    parsers['md'].add_argument('--temp0', type=float, default=300.0)
    parsers['md'].add_argument('--dt', type=float, default=0.002)
    parsers['md'].add_argument('--ntpr', type=int, default=2500)
    parsers['md'].add_argument('--ntwx', type=int, default=2500)
    parsers['md'].add_argument('--ntb', type=int, default=1)
    parsers['md'].add_argument('--pres0', type=float, default=1.0)

    parsers['webmd_prep'] = subparsers.add_parser('webmd_prep')
    parsers['webmd_prep'].add_argument('work_dir')
    parsers['webmd_prep'].add_argument('--buffer_size', default=10.0, type=float)
    parsers['webmd_prep'].add_argument('--forcefield', default='ff99SBildn')
    parsers['webmd_prep'].add_argument('--watermodel', default='TIP3PBOX')
    parsers['webmd_prep'].add_argument('--cation', default='Na+')
    parsers['webmd_prep'].add_argument('--anion', default='Cl-')

    parsers['webmd_min'] = subparsers.add_parser('webmd_min')
    parsers['webmd_min'].add_argument('work_dir')
    parsers['webmd_min'].add_argument('--maxcyc1', type=int, default=1000)
    parsers['webmd_min'].add_argument('--maxcyc2', type=int, default=2500)

    parsers['webmd_eq'] = subparsers.add_parser('webmd_eq')
    parsers['webmd_eq'].add_argument('work_dir')
    parsers['webmd_eq'].add_argument('-n1', '--nstlim1', type=int, default=10000)
    parsers['webmd_eq'].add_argument('-n2', '--nstlim2', type=int, default=100000)
    parsers['webmd_eq'].add_argument('--tempi', type=float, default=0.0)
    parsers['webmd_eq'].add_argument('--temp0', type=float, default=300.0)

    parsers['webmd_md'] = subparsers.add_parser('webmd_md')
    parsers['webmd_md'].add_argument('work_dir')
    parsers['webmd_md'].add_argument('-n', '--nstlim', type=int, default=500000)
    parsers['webmd_md'].add_argument('--temp0', type=float, default=300.0)
    parsers['webmd_md'].add_argument('--ntb', type=int, default=2)
    parsers['webmd_md'].add_argument('--pres0', type=float, default=1.0)

    parsers['anal'] = subparsers.add_parser('anal')
    anal_subparsers = parsers['anal'].add_subparsers(dest='method')
    anal_subparsers_map = dict()
    anal_subparsers_map['rmsd'] = anal_subparsers.add_parser('rmsd')
    anal_subparsers_map['rmsf'] = anal_subparsers.add_parser('rmsf')
    anal_subparsers_map['radgyr'] = anal_subparsers.add_parser('radgyr')
    anal_subparsers_map['sasa'] = anal_subparsers.add_parser('sasa')
    anal_subparsers_map['gsolv'] = anal_subparsers.add_parser('gsolv')
    anal_subparsers_map['eu'] = anal_subparsers.add_parser('eu')

    anal_subparsers_map['rmsd'].add_argument('reference_file')
    anal_subparsers_map['rmsd'].add_argument('topology_file')
    anal_subparsers_map['rmsd'].add_argument('out_file')
    anal_subparsers_map['rmsd'].add_argument('--mask', default='protein and name CA')
    anal_subparsers_map['rmsd'].add_argument('--trajin', nargs='+')

    anal_subparsers_map['rmsf'].add_argument('reference_file')
    anal_subparsers_map['rmsf'].add_argument('topology_file')
    anal_subparsers_map['rmsf'].add_argument('out_file')
    anal_subparsers_map['rmsf'].add_argument('--mask', default='protein and name CA')
    anal_subparsers_map['rmsf'].add_argument('--trajin', nargs='+')

    anal_subparsers_map['radgyr'].add_argument('reference_file')
    anal_subparsers_map['radgyr'].add_argument('topology_file')
    anal_subparsers_map['radgyr'].add_argument('out_file')
    anal_subparsers_map['radgyr'].add_argument('--mask', default='protein')
    anal_subparsers_map['radgyr'].add_argument('--trajin', nargs='+')

    anal_subparsers_map['sasa'].add_argument('reference_file')
    anal_subparsers_map['sasa'].add_argument('topology_file')
    anal_subparsers_map['sasa'].add_argument('out_file')
    anal_subparsers_map['sasa'].add_argument('--mask', default='protein')
    anal_subparsers_map['sasa'].add_argument('--trajin', nargs='+')

    anal_subparsers_map['eu'].add_argument('reference_file')
    anal_subparsers_map['eu'].add_argument('topology_file')
    anal_subparsers_map['eu'].add_argument('out_file')
    anal_subparsers_map['eu'].add_argument('--mask', default='protein')
    anal_subparsers_map['eu'].add_argument('--forcefield', default='ff99SBildn')
    anal_subparsers_map['eu'].add_argument('--trajin', nargs='+')

    anal_subparsers_map['gsolv'].add_argument('reference_file')
    anal_subparsers_map['gsolv'].add_argument('topology_file')
    anal_subparsers_map['gsolv'].add_argument('out_file')
    anal_subparsers_map['gsolv'].add_argument('--mask', default='protein')
    anal_subparsers_map['gsolv'].add_argument('--trajin', nargs='+')

    parsers['prowave'] = subparsers.add_parser('prowave')
    parsers['prowave'].add_argument('work_dir')
    parsers['prowave'].add_argument('mode')
    parsers['prowave'].add_argument('--box_size', type=float, default=128.0)
    parsers['prowave'].add_argument('--grid_size', type=int, default=128)
    parsers['prowave'].add_argument('--forcefield', default='ff99SBildn')

    parsers['rism3dx'] = subparsers.add_parser('rism3dx')
    parsers['rism3dx'].add_argument('rism_input_file')

    args = parser.parse_args()

    if args.func == 'prep':
        run_prep(args.pdb_source, args.pdb_target, args.topology_target, args.buffer_size,
                 force_field_name=args.forcefield, water_model_name=args.watermodel,
                 cation=args.cation, anion=args.anion)
    elif args.func == 'min':
        run_min(args.topology_source, args.pdb_source, args.maxcyc, args.state_file,
                restraint_weight=args.restraint_weight, reference=args.reference)
    elif args.func == 'eq':
        run_eq(args.topology_source, args.pdb_source, args.nstlim, args.state_file, args.out_file, args.reference,
               temp0=args.temp0, tempi=args.tempi, dt=args.dt, ntpr=args.ntpr, ntb=args.ntb,
               restraint_weight=args.restraint_weight)
    elif args.func == 'md':
        run_md(args.topology_source, args.pdb_source, args.nstlim, args.out_file, args.traj_file, args.state_file,
               args.reference, temp0=args.temp0, dt=args.dt, ntpr=args.ntpr, ntwx=args.ntwx, ntb=args.ntb,
               pres0=args.pres0)
    elif args.func == 'webmd_prep':
        webmd_prep(args.work_dir, buffer_size=args.buffer_size, ff_name=args.forcefield, wm_name=args.watermodel,
                   cation=args.cation, anion=args.anion)
    elif args.func == 'webmd_min':
        webmd_min(args.work_dir, args.maxcyc1, args.maxcyc2)
    elif args.func == 'webmd_eq':
        webmd_eq(args.work_dir, args.nstlim1, args.nstlim2, tempi=args.tempi, temp0=args.temp0)
    elif args.func == 'webmd_md':
        webmd_md(args.work_dir, args.nstlim, temp0=args.temp0, ntb=args.ntb, pres0=args.pres0)

    elif args.func == 'anal' and args.method == 'rmsd':
        calculate_rmsd(args.reference_file, args.topology_file, args.out_file, mask=args.mask, trajin=args.trajin)
    elif args.func == 'anal' and args.method == 'rmsf':
        calculate_rmsf(args.reference_file, args.topology_file, args.out_file, mask=args.mask, trajin=args.trajin)
    elif args.func == 'anal' and args.method == 'radgyr':
        calculate_radgyr(args.reference_file, args.topology_file, args.out_file, mask=args.mask, trajin=args.trajin)
    elif args.func == 'anal' and args.method == 'sasa':
        calculate_sasa(args.reference_file, args.topology_file, args.out_file, mask=args.mask, trajin=args.trajin)
    elif args.func == 'anal' and args.method == 'eu':
        calculate_eu(args.reference_file, args.topology_file, args.out_file, forcefield=args.forcefield,
                     mask=args.mask, trajin=args.trajin)
    elif args.func == 'anal' and args.method == 'gsolv':
        webmd_calculate_gsolv(args.reference_file, args.topology_file, args.out_file, mask=args.mask,
                              trajin=args.trajin)
    elif args.func == 'prowave':
        prowave_calcuate_gsolv(args.work_dir, args.mode, args.box_size, args.grid_size, args.forcefield)

    elif args.func == 'rism3dx':
        run_rism3dx(args.rism_input_file)

    else:
        print('Usage: smmd [ prep | min | eq | md | webmd_prep | webmd_min | webmd_eq | webmd_md | anal ]')
