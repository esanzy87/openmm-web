import os
import time
from django.http.response import HttpResponse
from django.shortcuts import render, redirect, reverse
from django.contrib import messages
from core.models import Topology


# Create your views here.
def root_view(request):
    return render(request, 'webmd/root.html')


def topologies_view(request):
    os.makedirs("topologies", exist_ok=True)
    topologies = []
    for path in os.listdir("topologies"):
        if os.path.isdir(f"topologies/{path}"):
            tm = time.localtime(os.path.getmtime(f"topologies/{path}"))
            topologies.append((Topology.load(path), time.strftime("%Y-%m-%d %H:%I:%S", tm)))
    return render(request, 'webmd/topologies.html', {
        'work_list': topologies,
    })


def new_topology_rcsb_view(request):
    if request.method == 'POST':
        topology = Topology(
            source_type=request.POST.get('source_type'),
            protein=request.POST.get('protein'),
            name=request.POST.get('name'),
            buffer_size=int(request.POST.get('buffer_size')),
            solvent_model=request.POST.get('solvent_model'),
            forcefield=request.POST.get('forcefield'),
        )
        topology.initialize()
        messages.success(request, 'New topology created successfully.')
        return redirect('/')
    return render(request, 'webmd/new_topology_rcsb.html')


def pre_process_select_model_and_chains(request):
    topology_name = request.GET.get('name')
    topology = Topology.load(topology_name)

    if request.method == 'POST':
        selected_model = int(request.POST['selected_model'])
        selected_chains = request.POST.getlist('selected_chains')
        if type(selected_chains) is not list:
            selected_chains = list(selected_chains)
        topology.select_model_and_chains(selected_model, selected_chains)
        topology.cleanup()
        return redirect(f'/topologies/pre-process/step-2/?name={topology_name}')

    with open(topology.model_pdb, 'r') as f:
        pdb_content = f.read()

    ctx = {'work': topology, 'pdb_content': pdb_content}
    return render(request, 'webmd/pre_process/select_model_and_chains.html', context=ctx)


def pre_process_convert_non_standard_residues(request):
    topology_name = request.GET.get('name')
    topology = Topology.load(topology_name)

    if request.POST:
        # build disulfide bonds
        cands = []
        cand_indices = request.POST.getlist('cands')
        for index in cand_indices:
            cand = topology.disulfide_bond_candidates[int(index)]
            r1 = cand[1]
            r2 = cand[2]

            cands.append((cand[0], r1.id))
            cands.append((cand[0], r2.id))
        topology.build_disulfide_bonds(cands)

        # mutate protonation states
        unknown_protonation_states = {
            (res.chain.id, res.id): request.POST.get(
                '%s %s %s' % (res.chain.id, res.id, res.name)) for res in topology.unknown_protonation_states
        }
        topology.mutate_protonation_states(unknown_protonation_states)

        return redirect(f'/topologies/pre-process/step-3/?name={topology_name}')

    with open(topology.work_dir / 'step2.pdb', 'r') as pdb_file:
        pdb_content = pdb_file.read()

    ctx = {'work': topology, 'pdb_content': pdb_content}
    return render(request, 'webmd/pre_process/convert_non_standard_residues.html', context=ctx)


def pre_process_done(request):
    topology_name = request.GET.get('name')
    topology = Topology.load(topology_name)

    if request.POST:
        cation = request.POST['cation']
        anion = request.POST['anion']

        topology.cation = cation
        topology.anion = anion
        topology.save()
        topology.pre_process_done(cation)
        return redirect('/topologies')

    with open(topology.work_dir / 'step2.pdb', 'r') as pdb_file:
        pdb_content = pdb_file.read()

    ctx = {
        'work': topology,
        'pdb_content': pdb_content
    }
    return render(request, 'webmd/pre_process/done.html', context=ctx)


def get_pdb_content_api(request):
    name = request.GET.get('name')
    step = request.GET.get('step', 'model')

    topology = Topology.load(name)

    try:
        with open(topology.work_dir / f'{step}.pdb', 'r') as pdb_file:
            return HttpResponse(pdb_file.read())
    except FileNotFoundError:
        return HttpResponse('')


def topology_structure_viewer(request):
    import gzip
    from pdbutil import PDBObject
    name = request.GET.get('name')
    topology = Topology.load(name)
    with gzip.open(topology.work_dir / 'prep/model_solv.pdb.gz', 'rt') as pdb_file:
        pdb_object = PDBObject.from_pdb_file(pdb_file)
    return render(request, 'webmd/structure_viewer.html', context={
        'pdb_content': pdb_object.pdb_string,
        'work': topology,
    })
