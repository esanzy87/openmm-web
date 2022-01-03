import os
import shutil
import pickle
import yaml
import pdbutil
import smmd
from typing import Optional
from pathlib import Path


# Create your models here.
class Topology:
    source_type: str
    protein: Optional[str]
    name: Optional[str]
    buffer_size: int
    solvent_model: str
    forcefield: str
    cation: Optional[str]
    anion: Optional[str]

    def __init__(self,
                 source_type='rcsb',
                 protein='1IYT',
                 name=None,
                 buffer_size=10,
                 solvent_model='TIP3PBOX',
                 forcefield='ff99SBildn',
                 cation='Na+',
                 anion='Cl-',
                 **kwargs):
        self.source_type = source_type
        self.protein = protein
        if not self.protein:
            self.protein = '1IYT'
        self.name = name
        if not self.name:
            self.name = self.protein
        self.buffer_size = buffer_size
        self.solvent_model = solvent_model
        self.forcefield = forcefield
        self.cation = cation
        self.anion = anion

    def dict(self):
        return {
            'source_type': self.source_type,
            'protein': self.protein,
            'name': self.name,
            'buffer_size': self.buffer_size,
            'solvent_model': self.solvent_model,
            'forcefield': self.forcefield,
            'cation': self.cation,
            'anion': self.anion,
        }

    @classmethod
    def load(cls, name):
        work_dir = Path(f'topologies/{name}')
        model_file_path = work_dir / 'model.yml'

        if not os.path.exists(model_file_path):
            raise AttributeError('Invalid topology name')

        with open(model_file_path, 'r') as schema_file:
            schema = yaml.load(schema_file, Loader=yaml.FullLoader)

        return cls(**schema)

    def save(self):
        model_file_path = self.work_dir / 'model.yml'
        with open(model_file_path, 'w') as schema_file:
            yaml.dump(self.dict(), schema_file)

    def initialize(self):
        import requests
        if os.path.exists(self.work_dir):
            raise AttributeError('Topology name already taken')

        response = requests.get('https://file.rcsb.org/download/%s.pdb' % self.protein.lower())
        if response.status_code == 404:
            raise AttributeError('RCSB code not valid')
        
        os.makedirs(self.work_dir)
        with open(self.work_dir / 'model.pdb', 'w') as f:
            f.write(response.content.decode())
        self.save()

    @property
    def work_dir(self):
        return Path(f'topologies/{self.name}')

    @property
    def model_pdb(self):
        return self.work_dir / 'model.pdb'

    @property
    def models(self):
        return pdbutil.get_models(self.model_pdb)

    @property
    def chains(self):
        return pdbutil.get_chains(self.model_pdb)

    @property
    def hetero_residues(self):
        return pdbutil.get_heterogenes(self.model_pdb)

    @property
    def non_standard_residues(self):
        return pdbutil.get_non_standard_residues(self.work_dir / 'step2.pdb')

    @property
    def disulfide_bond_candidates(self):
        return pdbutil.get_disulfide_bond_candidates(self.work_dir / 'step2.pdb')

    @property
    def solvent_ions(self):
        return pdbutil.get_solvent_ions(self.work_dir / 'step2.pdb')

    @property
    def unknown_protonation_states(self):
        return pdbutil.get_unkown_protonation_states(self.work_dir / 'step2.pdb')

    @property
    def prepared(self):
        return os.path.exists(self.work_dir / 'prep/model.xml')

    def select_model_and_chains(self, selected_model, selected_chains):
        shutil.copy2(self.work_dir / 'model.pdb', self.work_dir / 'step1.pdb')
        pdbutil.select_model_and_chains(self.work_dir / 'step1.pdb', selected_model, selected_chains)

    def cleanup(self):
        shutil.copy2(os.path.join(self.work_dir, 'step1.pdb'), os.path.join(self.work_dir, 'step2.pdb'))
        pdbutil.cleanup(self.work_dir / 'step2.pdb')

    def build_disulfide_bonds(self, cands):
        pass

    def mutate_protonation_states(self, protonation_states):
        with open(os.path.join(self.work_dir, 'protonation_states.pickle'), 'wb') as f:
            pickle.dump(protonation_states, f)

    def delete_solvent_ions(self, keeping_solvent_ions):
        with open(os.path.join(self.work_dir, 'keeping_solvent_ions.pickle'), 'wb') as f:
            pickle.dump(keeping_solvent_ions, f)

    def pre_process_done(self, cation):
        pdbutil.convert_non_standard_residues(self.work_dir / 'step2.pdb')

        try:
            with open(os.path.join(self.work_dir, 'keeping_solvent_ions.pickle'), 'rb') as f:
                keeping_solvent_ions = pickle.load(f)
            os.remove(os.path.join(self.work_dir, 'keeping_solvent_ions.pickle'))
        except FileNotFoundError:
            keeping_solvent_ions = set()  # an empty set, delete all solvent ions.
        pdbutil.keep_solvent_ions(self.work_dir / 'step2.pdb', keeping_solvent_ions)

        try:
            with open(os.path.join(self.work_dir, 'protonation_states.pickle'), 'rb') as f:
                protonation_states = pickle.load(f)
            os.remove(os.path.join(self.work_dir, 'disulfide_bond_candidates.pickle'))
        except FileNotFoundError:
            protonation_states = {}
        variants = pdbutil.determine_protonation_states(self.work_dir / 'step2.pdb', protonation_states)
        pdbutil.add_missing_hydrogens(self.work_dir / 'step2.pdb', variants=variants)

        shutil.copy2(self.work_dir / 'step2.pdb', self.work_dir / 'model.pdb')
        os.remove(self.work_dir / 'step1.pdb')
        os.remove(self.work_dir / 'step2.pdb')

        # self.cation = cation
        # self.save()
        smmd.webmd_prep(
            self.work_dir,
            buffer_size=10,
            ff_name='ff99SBildn',
            wm_name='TIP3PBOX',
            cation='Na+',  # cation=cation,
            anion='Cl-',
        )


class Simulation:
    name: Optional[str]
    topology: Optional[Topology]
    minimizations: Optional[list]
    equilibrations: Optional[list]
    productions: Optional[list]

    def __init__(self, name, topology=None, minimizations=None, equilibrations=None, productions=None):
        self.name = name
        self.topology = None
        if topology:
            self.topology = topology
        self.minimizations = []
        if minimizations:
            self.minimizations = minimizations
        self.equilibrations = []
        if equilibrations:
            self.equilibrations = equilibrations
        self.productions = []
        if productions:
            self.productions = productions

    @property
    def work_dir(self):
        return Path(f'simulations/{self.name}')

    def dict(self):
        return {
            'name': self.name,
            'topology': self.topology.dict(),
            'minimizations': self.minimizations,
            'equilibrations': self.equilibrations,
            'productions': self.productions,
        }

    @classmethod
    def load(cls, name):
        work_dir = Path(f'simulations/{name}')
        protocols_file_path = work_dir / 'protocols.yml'

        if not os.path.exists(protocols_file_path):
            raise AttributeError('Invalid simulation name')

        with open(protocols_file_path, 'r') as protocols_file:
            protocol = yaml.load(protocols_file, Loader=yaml.FullLoader)

        return cls(
            name=protocol['name'],
            topology=Topology.load(name=protocol['topology']['name']),
            minimizations=protocol['minimizations'],
            equilibrations=protocol['equilibrations'],
            productions=protocol['productions'],
        )
    
    def initialize(self, topology_name):
        if os.path.exists(self.work_dir):
            raise AttributeError('Simulation name already taken')

        topology_file_path = f'topologies/{topology_name}/prep/model.xml'
        if not os.path.exists(topology_file_path):
            raise AttributeError('Invalid topology name')

        os.makedirs(self.work_dir, exist_ok=True)
        shutil.copytree(f'topologies/{topology_name}/prep', self.work_dir / 'prep')
        
        self.topology = Topology.load(name=topology_name)
        self.add_minimization(1000, 10.0)
        self.add_minimization(2500)
        self.add_equilibration(10000, restraint_weight=10.0)
        self.add_equilibration(100000)
        self.add_production(100000)
        self.add_production(100000)
        self.add_production(100000)
        self.add_production(100000)
        self.save()

    def save(self):
        protocols_file_path = self.work_dir / 'protocols.yml'
        with open(protocols_file_path, 'w') as protocols_file:
            yaml.dump(self.dict(), protocols_file)

    def add_minimization(self, steps, restraint_weight=None):
        self.minimizations.append({
            'steps': steps,
            'restraint_weight': restraint_weight,
        })
        self.save()
    
    def remove_last_minimization(self):
        idx = len(self.minimizations)
        self.minimizations = self.minimizations[:-1]
        shutil.rmtree(self.work_dir / f'min/{idx}', ignore_errors=True)
        self.save()
    
    def add_equilibration(self, steps, initial_temperature=0.0, reference_temperature=300, step_size=0.002, ensemble='NVT', restraint_weight=None):
        self.equilibrations.append({
            'steps': steps,
            'initial_temperature': initial_temperature,
            'reference_temperature': reference_temperature,
            'step_size': step_size,
            'ensemble': ensemble,
            'restraint_weight': restraint_weight,
        })
        self.save()

    def remove_last_equilibration(self):
        idx = len(self.equilibrations)
        self.equilibrations = self.equilibrations[:-1]
        shutil.rmtree(self.work_dir / f'eq/{idx}', ignore_errors=True)
        self.save()

    def add_production(self, steps, reference_temperature=300, reference_pressure=None, step_size=0.002, ensemble='NVT'):
        self.productions.append({
            'steps': steps,
            'reference_temperature': reference_temperature,
            'reference_pressure': reference_pressure,
            'step_size': step_size,
            'ensemble': ensemble,
        })
        self.save()

    def remove_last_production(self):
        idx = len(self.productions)
        self.productions = self.productions[:-1]
        shutil.rmtree(self.work_dir / f'md/{idx}', ignore_errors=True)
        self.save()

    def run(self):
        for idx, min in enumerate(self.minimizations, 1):
            if os.path.exists(self.work_dir / f'min/{idx}/min.npz'):
                continue
            os.makedirs(self.work_dir / f'min/{idx}', exist_ok=True)
            if idx == 1:
                t, st = smmd.run_min(
                    pdb_source=str(self.work_dir / 'prep/model_solv.pdb'),
                    topology_source=str(self.work_dir / 'prep/model.xml'),
                    maxcyc=min['steps'],
                    state_file=self.work_dir / f'min/{idx}/min.npz',
                    restraint_weight=min['restraint_weight'],
                )
            else:
                t, st = smmd.run_min(
                    pdb_source=str(self.work_dir / 'prep/model_solv.pdb'),
                    topology_source=str(self.work_dir / 'prep/model.xml'),
                    maxcyc=min['steps'],
                    state_file=self.work_dir / f'min/{idx}/min.npz',
                    restraint_weight=min['restraint_weight'],
                    reference=self.work_dir / f'min/{idx-1}/min.npz'
                )
        
        for idx, eq in enumerate(self.equilibrations, 1):
            if os.path.exists(self.work_dir / f'eq/{idx}/eq.npz'):
                continue
            os.makedirs(self.work_dir / f'eq/{idx}', exist_ok=True)
            if idx == 1:
                t, st = smmd.run_eq(
                    pdb_source=str(self.work_dir / 'prep/model_solv.pdb'),
                    topology_source=str(self.work_dir / 'prep/model.xml'),
                    nstlim=eq['steps'],
                    dt=eq['step_size'],
                    out_file=str(self.work_dir / f'eq/{idx}/eq.out'),
                    state_file=self.work_dir / f'eq/{idx}/eq.npz',
                    reference=self.work_dir / f'min/{len(self.minimizations)}/min.npz',
                    tempi=eq['initial_temperature'],
                    temp0=eq['reference_temperature'],
                    ntb=1 if eq['ensemble'] == 'NVT' else 2,
                    restraint_weight=eq['restraint_weight'],
                )
            else:
                t, st = smmd.run_eq(
                    pdb_source=str(self.work_dir / 'prep/model_solv.pdb'),
                    topology_source=str(self.work_dir / 'prep/model.xml'),
                    nstlim=eq['steps'],
                    dt=eq['step_size'],
                    out_file=str(self.work_dir / f'eq/{idx}/eq.out'),
                    state_file=self.work_dir / f'eq/{idx}/eq.npz',
                    reference=self.work_dir / f'eq/{idx-1}/eq.npz',
                    temp0=eq['reference_temperature'],
                    ntb=1 if eq['ensemble'] == 'NVT' else 2,
                    restraint_weight=eq['restraint_weight'],
                )
            
        for idx, md in enumerate(self.productions, 1):
            if os.path.exists(self.work_dir / f'eq/{idx}/md.dcd'):
                continue
            os.makedirs(self.work_dir / f'md/{idx}', exist_ok=True)
            if idx == 1:
                t, st = smmd.run_md(
                    pdb_source=str(self.work_dir / 'prep/model_solv.pdb'),
                    topology_source=str(self.work_dir / 'prep/model.xml'),
                    nstlim=md['steps'],
                    dt=md['step_size'],
                    out_file=str(self.work_dir / f'md/{idx}/md.out'),
                    traj_file=self.work_dir / f'md/{idx}/md.dcd',
                    state_file=self.work_dir / f'md/{idx}/md.npz',
                    reference=self.work_dir / f'eq/{len(self.equilibrations)}/eq.npz',
                    temp0=md['reference_temperature'],
                    ntb=1 if md['ensemble'] == 'NVT' else 2,
                    pres0=md['reference_pressure'],
                )
            else:
                t, st = smmd.run_md(
                    pdb_source=str(self.work_dir / 'prep/model_solv.pdb'),
                    topology_source=str(self.work_dir / 'prep/model.xml'),
                    nstlim=md['steps'],
                    dt=md['step_size'],
                    out_file=str(self.work_dir / f'md/{idx}/md.out'),
                    traj_file=self.work_dir / f'md/{idx}/md.dcd',
                    state_file=self.work_dir / f'md/{idx}/md.npz',
                    reference=self.work_dir / f'md/{idx-1}/md.npz',
                    temp0=md['reference_temperature'],
                    ntb=1 if md['ensemble'] == 'NVT' else 2,
                    pres0=md['reference_pressure'],
                )

    #
    # def run_min(self, maxcyc1, maxcyc2):
    #     from utils.smmd import submit_batch, webmd_min
    #     job_id = submit_batch(os.path.join(self.work_dir, 'min'), 'webmd_min',
    #                           partition=settings.SLURM_WEBMD_PARTITION,
    #                           work_dir=self.work_dir, maxcyc1=maxcyc1, maxcyc2=maxcyc2)
    #
    #     if job_id != '-1':
    #         WorkJob.objects.create(job_id=job_id, work_id=self.id)
    #         return job_id
    #     else:
    #         webmd_min(self.work_dir, maxcyc1, maxcyc2)
    #         return None
    #
    # def run_eq(self, init_temp, ref_temp, nstlim1, nstlim2):
    #     from utils.smmd import submit_batch, webmd_eq
    #
    #     job_id = submit_batch(os.path.join(self.work_dir, 'eq'), 'webmd_eq',
    #                           partition=settings.SLURM_WEBMD_PARTITION,
    #                           work_dir=self.work_dir, nstlim1=nstlim1, nstlim2=nstlim2, tempi=init_temp, temp0=ref_temp)
    #
    #     if job_id != '-1':
    #         WorkJob.objects.create(job_id=job_id, work_id=self.id)
    #         return job_id
    #     else:
    #         webmd_eq(self.work_dir, nstlim1, nstlim2, init_temp, ref_temp)
    #         return None
    #
    # def run_md(self, md_serial, ref_temp, ntb, nstlim, pressure, dependency=None):
    #     from utils.smmd import submit_batch, webmd_md
    #
    #     job_id = submit_batch(os.path.join(self.work_dir, 'md/%d' % md_serial), 'webmd_md',
    #                           dependency=dependency, partition=settings.SLURM_WEBMD_PARTITION,
    #                           work_dir=self.work_dir, nstlim=nstlim, temp0=ref_temp, ntb=ntb, pres0=pressure)
    #
    #     if job_id != '-1':
    #         WorkJob.objects.create(job_id=job_id, work_id=self.id)
    #         return job_id
    #     else:
    #         if dependency is None:
    #             webmd_md(self.work_dir, nstlim, ref_temp, ntb, pressure)
    #         return None
    #
    # def create_analysis(self, method, trajin, mask=None):
    #     from utils.smmd import submit_batch
    #     analyses = sorted([serial for serial in self.get_analyses().keys()])
    #     new_serial = analyses[-1] + 1 if analyses else 1
    #
    #     base_dir = os.path.join(self.work_dir, 'analyses/%d' % new_serial)
    #     try:
    #         os.makedirs(base_dir)
    #     except OSError:
    #         pass
    #
    #     reference_file = os.path.join(self.work_dir, 'prep/model_solv.pdb.gz')
    #     topology_file = os.path.join(self.work_dir, 'prep/model_solv.pdb.gz')
    #
    #     with open(os.path.join(base_dir, '%s.out' % method), 'w') as f:
    #         f.write('\n')
    #
    #     if method == 'eu':
    #         job_id = submit_batch(base_dir, 'anal', partition=settings.SLURM_WEBMD_PARTITION,
    #                               method=method, mask=mask or 'protein',
    #                               reference_file=reference_file, topology_file=topology_file,
    #                               out_file=os.path.join(base_dir, 'eu.out'),
    #                               trajin=[os.path.join(self.work_dir, 'md/%s/md.dcd' % i) for i in trajin],
    #                               forcefield='ff99SBildn')
    #     else:
    #         job_id = submit_batch(base_dir, 'anal', partition=settings.SLURM_WEBMD_PARTITION,
    #                               method=method, mask=mask or 'protein',
    #                               reference_file=reference_file, topology_file=topology_file,
    #                               out_file=os.path.join(base_dir, '%s.out' % method),
    #                               trajin=[os.path.join(self.work_dir, 'md/%s/md.dcd' % i) for i in trajin])
    #
    #     if job_id != '-1':
    #         WorkJob.objects.create(job_id=job_id, work_id=self.id)
    #         WorkAnalysisJob.objects.create(job_id=job_id, work=self, anal_serial=new_serial)
    #         return job_id
    #     else:
    #         return None
    #
    # def get_traj(self, md_serial):
    #     dcd_path = os.path.join(self.work_dir, 'md/{}/md.dcd'.format(md_serial))
    #     return dcd_path, open(dcd_path, 'rb')
    #
    # def _load_whole_traj(self):
    #     r = mdt.load(os.path.join(self.work_dir, 'prep/model_solv.pdb.gz'))
    #     sel = r.topology.select('protein')
    #     ts = []
    #     for serial in self.completed_md_serials:
    #         traj = mdt.load(os.path.join(self.work_dir, 'md/%d/md.dcd' % serial),
    #                         top=os.path.join(self.work_dir, 'prep/model_solv.pdb.gz'))
    #         traj.image_molecules(inplace=True)
    #         traj.center_coordinates()
    #         traj.topology = traj.topology.subset(sel)
    #         traj.xyz = traj.xyz[:, sel, :]
    #         ts.append(traj)
    #
    #     if len(ts) > 1:
    #         trajs = ts[0].join(ts[1:])
    #     else:
    #         trajs = ts[0]
    #
    #     return trajs
    #
    # def get_traj_info(self, md_serial):
    #     if md_serial != 0:
    #         dcd_path = os.path.join(self.work_dir, 'md/%s/md.dcd' % md_serial)
    #         t = mdt.load(dcd_path, top=os.path.join(self.work_dir, 'prep/model_solv.pdb.gz'))
    #     else:
    #         dcd_path = os.path.join(self.work_dir, 'md/md.dcd')
    #         t = self._load_whole_traj()
    #         t.save(dcd_path)
    #
    #     return {
    #         'frames': t.n_frames,
    #         'atoms': t.n_atoms,
    #         'residues': t.n_residues,
    #         'size': os.path.getsize(dcd_path),
    #     }
    #
    # def get_whole_traj(self):
    #     dcd_path = os.path.join(self.work_dir, 'md/md.dcd')
    #     return dcd_path, open(dcd_path, 'rb')
    #
    # def get_pdb_at_frame(self, md_serial, frame):
    #     dcd_path = os.path.join(self.work_dir, 'md/%s/md.dcd' % md_serial)
    #     t = mdt.load_frame(dcd_path, int(frame), top=os.path.join(self.work_dir, 'prep/model_solv.pdb.gz'))
    #     t.image_molecules(inplace=True)
    #     t.center_coordinates()
    #     pdb_path = os.path.join(self.work_dir, 'md/{}/frame_{}.pdb'.format(md_serial, frame))
    #     t.save_pdb(pdb_path, force_overwrite=True)
    #
    #     with open(pdb_path, 'r') as f:
    #         pdb_content = f.read()
    #
    #     os.remove(pdb_path)
    #     return pdb_content
    #
    # def get_trajectory_at_frame(self, md_serial, frame):
    #     dcd_path = os.path.join(self.work_dir, 'md/%s/md.dcd' % md_serial)
    #     t = mdt.load_frame(dcd_path, int(frame), top=os.path.join(self.work_dir, 'prep/model_solv.pdb.gz'))
    #     sel = t.topology.select('protein')
    #     t.image_molecules()
    #     t.topology = t.topology.subset(sel)
    #     t.xyz = t.xyz[:, sel, :]
    #     t.center_coordinates(mass_weighted=False)
    #     return t
    #
    # def get_batch_job_id(self, step):
    #     if step in ('prep', 'min', 'eq'):
    #         with open(os.path.join(self.work_dir, '%s/status' % step), 'r') as f:
    #             status = f.read().strip()
    #     elif 'md' in step:
    #         md_serial = int(step[2:])
    #         with open(os.path.join(self.work_dir, 'md/%d/status' % md_serial), 'r') as f:
    #             status = f.read().strip()
    #     else:
    #         return None
    #     try:
    #         return int(status.split()[1])
    #     except:
    #         return None
    #
    # def delete(self, using=None, keep_parents=False):
    #     try:
    #         shutil.rmtree(self.work_dir)
    #     except FileNotFoundError:
    #         pass
    #     super(Work, self).delete(using, keep_parents)
    #
    # def get_analyses(self):
    #     analyses_root_dir = os.path.join(self.work_dir, 'analyses')
    #
    #     if not os.path.exists(analyses_root_dir):
    #         return dict()
    #
    #     anal_serial = sorted([int(serial) for serial in next(os.walk(analyses_root_dir))[1]])
    #     ret = dict()
    #     for serial in anal_serial:
    #         ret[serial] = [None, None, 'Error']
    #
    #         base_dir = os.path.join(analyses_root_dir, '%d' % serial)
    #
    #         if os.path.exists(os.path.join(base_dir, 'rmsd.out')):
    #             output_file = os.path.join(base_dir, 'rmsd.out')
    #             anal_method = 'rmsd'
    #         elif os.path.exists(os.path.join(base_dir, 'rmsf.out')):
    #             output_file = os.path.join(base_dir, 'rmsf.out')
    #             anal_method = 'rmsf'
    #         elif os.path.exists(os.path.join(base_dir, 'radgyr.out')):
    #             output_file = os.path.join(base_dir, 'radgyr.out')
    #             anal_method = 'radgyr'
    #         elif os.path.exists(os.path.join(base_dir, 'sasa.out')):
    #             output_file = os.path.join(base_dir, 'sasa.out')
    #             anal_method = 'sasa'
    #         elif os.path.exists(os.path.join(base_dir, 'eu.out')):
    #             output_file = os.path.join(base_dir, 'eu.out')
    #             anal_method = 'eu'
    #         elif os.path.exists(os.path.join(base_dir, 'gsolv.out')):
    #             output_file = os.path.join(base_dir, 'gsolv.out')
    #             anal_method = 'gsolv'
    #         else:
    #             return ret
    #
    #         try:
    #             with open(output_file, 'r') as f:
    #                 output = [line.split() for line in f.readlines()[1:] if line]
    #
    #             ret[serial] = [anal_method, json.dumps(output), 'Done']
    #         except FileNotFoundError:
    #             ret[serial] = [anal_method, '', 'Error']
    #
    #         job_ids = running_job_ids()
    #         for obj in WorkAnalysisJob.objects.filter(work=self, anal_serial=serial).all():
    #             if obj.job_id in job_ids:
    #                 ret[serial][2] = 'Running'
    #
    #     return ret
    #
    #
    #
    # @property
    # def status(self):
    #     prep_status = os.path.join(self.work_dir, 'prep/status')
    #     model_solv_pdb = os.path.join(self.work_dir, 'prep/model_solv.pdb.gz')
    #     min_status = os.path.join(self.work_dir, 'min/status')
    #     min2_npz = os.path.join(self.work_dir, 'min/min2.npz')
    #     eq_status = os.path.join(self.work_dir, 'eq/status')
    #     eq2_npz = os.path.join(self.work_dir, 'eq/eq2.npz')
    #
    #     if self.md_serials:
    #         if len(self.md_serials) == len(self.completed_md_serials):
    #             return {'key': 'analysis_ready', 'display': 'Ready to run analysis', 'enum': 8}
    #
    #         md_serial = sorted(list(set(self.md_serials) - set(self.completed_md_serials)))[0]
    #         md_status = os.path.join(self.work_dir, 'md/%d/status' % md_serial)
    #         # md_npz = os.path.join(self.work_dir, 'md/%d/md.npz' % md_serial)
    #
    #         if os.path.exists(md_status):
    #             job_id = self.get_batch_job_id('md%d' % md_serial)
    #             if job_id in running_job_ids():
    #                 return {'key': 'md_running',
    #                         'display': 'MD (%d/%d) running ...' % (md_serial, len(self.md_serials)), 'enum': 7}
    #             else:
    #                 return {'key': 'md_error', 'display': 'MD #%d error' % md_serial, 'enum': 6.5}
    #
    #     if os.path.exists(eq2_npz):
    #         return {'key': 'md_ready', 'display': 'Ready to run MD simulation', 'enum': 6}
    #
    #     if os.path.exists(eq_status):
    #         job_id = self.get_batch_job_id('eq')
    #         if job_id in running_job_ids():
    #             return {'key': 'eq_running', 'display': 'EQ running ...', 'enum': 5}
    #         else:
    #             return {'key': 'eq_error', 'display': 'EQ error', 'enum': 4.5}
    #
    #     if os.path.exists(min2_npz):
    #         return {'key': 'eq_ready', 'display': 'Ready to run EQ', 'enum': 4}
    #
    #     if os.path.exists(min_status):
    #         job_id = self.get_batch_job_id('min')
    #         if job_id in running_job_ids():
    #             return {'key': 'min_running', 'display': 'MIN running ...', 'enum': 3}
    #         else:
    #             return {'key': 'min_error', 'display': 'MIN error', 'enum': 2.5}
    #
    #     if os.path.exists(model_solv_pdb):
    #         return {'key': 'min_ready', 'display': 'Ready to run MIN', 'enum': 2}
    #
    #     if os.path.exists(prep_status):
    #         job_id = self.get_batch_job_id('prep')
    #         if job_id in running_job_ids():
    #             return {'key': 'prep_running', 'display': 'PREP running ...', 'enum': 1}
    #         else:
    #             return {'key': 'prep_error', 'display': 'PREP error', 'enum': 0.5}
    #
    #     return {'key': 'pre_process_ready', 'display': 'Ready to pre process', 'enum': 0}
    #
    # @property
    # def md_serials(self):
    #     if os.path.exists(os.path.join(self.work_dir, 'md')):
    #         return sorted([int(serial) for serial in next(os.walk(os.path.join(self.work_dir, 'md')))[1]])
    #     else:
    #         return []
    #
    # @property
    # def completed_md_serials(self):
    #     md_root_dir = os.path.join(self.work_dir, 'md')
    #     if os.path.exists(md_root_dir):
    #         ret = []
    #         for serial in next(os.walk(md_root_dir))[1]:
    #             md_npz = os.path.join(md_root_dir, '%s/md.npz' % serial)
    #             if os.path.exists(md_npz):
    #                 ret.append(int(serial))
    #         return sorted(ret)
    #     else:
    #         return []
    #
    # @property
    # def sequence(self):
    #     seq = dict()
    #     for chain in self.chains:
    #         cmd = "cat %s | awk '/ATOM/ && $3 == \"CA\" && $5 == \"%s\" {print $4}' | tr '\n' ' ' | sed 's/ALA/A/g;s/CYS/C/g;s/ASP/D/g;s/GLU/E/g;s/PHE/F/g;s/GLY/G/g;s/HIS/H/g;s/ILE/I/g;s/LYS/K/g;s/LEU/L/g;s/MET/M/g;s/ASN/N/g;s/PRO/P/g;s/GLN/Q/g;s/ARG/R/g;s/SER/S/g;s/THR/T/g;s/VAL/V/g;s/TRP/W/g;s/TYR/Y/g' | sed 's/ //g' | fold -w 60" % (os.path.join(self.work_dir, 'model.pdb'), chain)
    #         seq[chain] = subprocess.check_output(cmd, shell=True).decode()
    #
    #     ret = ''
    #     for chain, s in seq.items():
    #         ret += '%s\n%s\n' % (chain, s)
    #     return ret
