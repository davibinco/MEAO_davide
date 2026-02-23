import tequila as tq
import sunrise as sun
import numpy as np
from scipy.stats import ortho_group
#need to add mscf, fci
from pyscf import gto, scf, mcscf, fci
from pyscf.lo.iao import iao
from pyscf.lo import orth
from functools import reduce
from MEAO_davide.meao import MEAO

# Build the molecule of interest
mol = gto.M(atom='N 0 0 0; N 0 0 1.094',
    spin=0, verbose=0,basis='sto-3g',unit='A',
    max_memory=1000,symmetry = False) # mem in MB

# Build the reference MINAO molecule
pmol = gto.M(atom='N 0 0 0; N 0 0 1.094',
    spin=0, verbose=0,basis='minao',unit='A',
    max_memory=1000,symmetry = False) # mem in MB

aoslices = pmol.aoslice_by_atom()
norbs_in_atoms = []
for ia in range(pmol.natm):
    norbs_in_atoms.append(aoslices[ia][3]-aoslices[ia][2])

# Run RHF calculation
mf = scf.RHF(mol)
mf.kernel()

#we add the state-average
n_states = 2
weights = np.ones(n_states)/n_states
mycas = mcscf.CASSCF(mf, 6,6)

mycas.fcisolver = fci.direct_spin0.FCI(mol)

mycas.state_average_(weights)

mycas.kernel()

ci = mycas.ci
ci1 = ci[1]
D_mo_ee = fci.direct_spin0.make_rdm1(ci1, mycas.ncas, mycas.nelecas)

mycas.mo_occ = np.diagonal(D_mo_ee)

# Construct IAOs
orbocc = mycas.mo_coeff[:,mycas.mo_occ>0]
c = iao(mol, orbocc)
s = mol.intor('int1e_ovlp')
mo_iao = np.dot(c, orth.lowdin(reduce(np.dot, (c.T,s,c))))

# Construct 1RDM in HF basis
#mo_hf = mf.mo_coeff
#dm1_hf = np.zeros((len(mo_hf),len(mo_hf)))
#dm1_hf[mf.mo_occ>0,mf.mo_occ>0] = 2

#we substitute the above part for:

mo_sa = mycas.mo_coeff
dm1_sa = D_mo_ee

# Construct 1RDM in IAO basis
U = reduce(np.dot, (mo_iao.T,s,mo_sa))
dm1_iao = U @ dm1_sa @ U.T

my_meao = MEAO(mol, mycas, mo_iao, norbs_in_atoms)
my_meao.meao()
bonds = my_meao.get_bonds()
print('Bonds:', bonds)
C_meao = my_meao.mo_meao
print(C_meao)

mol = sun.Molecule(
        geometry="N 0 0 0\nN 0 0 1.094",
        basis_set="sto-3g"
    )
mol.integral_manager.orbital_coefficients = C_meao

print("\nPlotting with Sunrise...")
sun.plot_MO(mol)