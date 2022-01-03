"""URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
# from django.contrib import admin
from django.urls import path
from core import views as core

urlpatterns = [
    # path('admin/', admin.site.urls),
    path('', core.root_view),
    path('get-pdb-content-api', core.get_pdb_content_api),
    path('topologies/', core.topologies_view),
    path('topologies/new/', core.new_topology_rcsb_view),
    path('topologies/pre-process/step-1/', core.pre_process_select_model_and_chains),
    path('topologies/pre-process/step-2/', core.pre_process_convert_non_standard_residues),
    path('topologies/pre-process/step-3/', core.pre_process_done),
    path('topologies/structure/', core.topology_structure_viewer),
    path('topologies/create-simulation/', core.make_simulation_from_topology),
    path('simulations/run/', core.simulation_run),
]
