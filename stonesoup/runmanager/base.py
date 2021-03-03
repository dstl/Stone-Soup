
print('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(__file__,__name__,str(__package__)))

from ..base import Base
from stonesoup.types.array import StateVector


class RunManager(Base):
    "Base run manager class"

    def __init__(self):
        self.state_vectors=[]
        self.state_vector_min_range=[]
        self.state_vector_max_range=[]
        self.state_vector_step=[]
        
        self.covar=[]
        self.covar_min_range=[]
        self.covar_max_range=[]
        self.covar_step=[]

        self.number_particles=[]
        self.number_particles_min_range=0
        self.number_particles_max_range=0
        self.number_particles_step=0