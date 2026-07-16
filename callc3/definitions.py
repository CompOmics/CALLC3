

available_atom_features = [
    ('Accessible surface area contribution (scalar)', 'accessible_surface_area_contribution'),
    ('Aromatic (boolean)', 'is_aromatic'),
    ('Atom type (nominal)', 'atom_type'),
    ('Atom weight (scalar)', 'atomic_weight'),
    ('CIP code (nominal)', 'c_i_p_code'),
    ('Chirality possible (boolean)', 'is_chirality_possible'),
    ('Degree (nominal)', 'degree'),
    ('Formal charge (nominal)', 'formal_charge'),
    ('Heteroatom (boolean)', 'is_heteroatom'),
    ('Hybridization (nominal)', 'hybridization'),
    ('Hydrogen acceptor (bool)', 'is_hydrogen_acceptor'),
    ('Hydrogen donor (boolean)', 'is_hydrogen_donor'),
    ('Log P contribution (scalar)', 'log_p_contribution'),
    ('Molar refractivity contribution (scalar)', 'molar_refractivity_contribution'),
    ('Number of hydrogens (nominal)', 'num_hydrogens'),
    ('Number of radical electrons (nominal)', 'num_radical_electrons'),
    ('Partial charge (nominal)', 'partial_charge'),
    ('Ring (boolean)', 'is_in_ring'),
    ('Ring size (nominal)', 'ring_size'),
    ('Total polar surface area contribution (scalar)', 'total_polar_surface_area_contribution'),
    ('Valence (nominal)', 'valence'),
]

available_bond_features = [
    ('Bond type (nominal)', 'bond_type'),
    ('Conjugated (boolean)', 'is_conjugated'),
    ('Rotatable (boolean)', 'is_rotatable'),
    ('Stereo (nominal)', 'stereo'),
]

available_molecule_features = [
    ('Accessible surface area (scalar)', 'accessible_surface_area'),
    ('Log P (scalar)', 'log_p'),
    ('Molar refractivity (scalar)', 'molar_refractivity'),
    ('Molecular weight (scalar)', 'mol_weight'),
    ('Number of aromatic rings (ordinal)', 'num_aromatic_rings'),
    ('Number of heavy atoms (ordinal)', 'num_heavy_atoms'),
    ('Number of heteroatoms (ordinal)', 'num_heteroatoms'),
    ('Number of hydrogen acceptors (ordinal)', 'num_hydrogen_acceptors'),
    ('Number of hydrogen donors (ordinal)', 'num_hydrogen_donors'),
    ('Number of rings (ordinal)', 'num_rings'),
    ('Number of rotatable bonds (ordinal)', 'num_rotatable_bonds'),
    ('Total polar surface area (scalar)', 'total_polar_surface_area'),
]

default_atom_features = {'atom_type', 'degree', 'num_hydrogens'}

default_bond_features = {'bond_type', 'is_rotatable'}

default_molecule_features = {'log_p', 'total_polar_surface_area'}

model_sizes = {
    'Small (64d)': 64,
    'Medium (128d)': 128,
    'Large (256d)': 256,
    'Extra Large (512d)': 512,
}

model_depths = {
    'Shallow (2r)': 2,
    'Medium (3r)': 3,
    'Deep (5r)': 5,
    'Very Deep (10r)': 10,
}

model_types = {
    'Convolutional': 'gi_conv',
    'Attentional': 'ga_conv',
    'Message Passing': 'mpnn_conv'
}

learning_rates = {
    'Very Low (1e-5)': 1e-5,
    'Low (1e-4)': 1e-4,
    'Medium (5e-4)': 5e-4,
    'High (1e-3)': 1e-3,
}

dropout_rates = {
    'None (0.0)': 0.0,
    'Low (0.1)': 0.1,
    'Medium (0.2)': 0.2,
    'High (0.3)': 0.3,
    'Very High (0.5)': 0.5,
}
