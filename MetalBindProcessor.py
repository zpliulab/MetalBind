import os
import _pickle as cPickle
import numpy as np
from sklearn.neighbors import KDTree
from subprocess import Popen, PIPE

from Bio.PDB import *
from Bio.Seq import Seq
from Bio import SeqIO
from Bio.SeqUtils import seq3

from default_config.bin_path import bin_path

from compute_surface.protonate import protonate
from compute_surface.extract_multimer import extractPDB
from compute_surface.extract_xyzrn import extract_xyzrn, atom_radii
from compute_surface.apply_msms import computeMSMS
from compute_surface.fixmesh import fix_mesh
from compute_surface.compute_normal import compute_normal

from compute_surface.get_protein_seq import get_seq
import pymesh

# pKa values from Lehninger Principles of Biochemistry 4th Ed.
pKa_scale = {
    "ILE": (2.36, 9.68, 7.0), "VAL": (2.32, 9.62, 7.0), "LEU": (2.36, 9.60, 7.0),
    "PHE": (1.83, 9.13, 7.0), "CYS": (1.96, 10.28, 8.18), "MET": (2.28, 9.21, 7.0),
    "ALA": (2.34, 9.69, 7.0), "GLY": (2.34, 9.60, 7.0), "THR": (2.11, 9.62, 7.0),
    "SER": (2.21, 9.15, 7.0), "TRP": (2.38, 9.39, 7.0), "TYR": (2.20, 9.11, 10.07),
    "PRO": (1.99, 10.96, 7.0), "HIS": (1.82, 9.17, 6.00), "GLU": (2.19, 9.67, 4.25),
    "GLN": (2.17, 9.13, 7.0), "ASP": (1.88, 9.60, 3.65), "ASN": (2.02, 8.80, 7.0),
    "LYS": (2.18, 8.95, 10.53), "ARG": (2.17, 9.04, 12.48)
}


class MetalBindProcessor():
    def __init__(self, pair, anno, ligand_type='', dir_opts={}, RQ_window=12.0):
        self.ligand_type = ligand_type
        self.dir_opts = dir_opts
        self.pair = pair
        self.RQ_window = RQ_window

        # parse identifiers
        self.pdb_id, self.protein_chains, self.nucleic_chains = pair.split(':')

        # generate surface mesh
        self.mesh, self.normalv, self.vert_info = self._get_surface()

        # extract sequence and mapping from get_seq
        self.seq, self.index2resid = get_seq(self.pdb_id, self.protein_chains, dir_opts=self.dir_opts)

        # prepare pKa mapping
        self.pKa_mapping = self._get_pKa_feature()

        # load ESM embeddings for each chain
        self.feature_esm = {}
        esm_dir = self.dir_opts.get('esm_dir', 'Dataset/600M')
        for chain in self.protein_chains.split('_'):
            esm_path = os.path.join(esm_dir, f"{self.pdb_id}{chain}.txt")
            if not os.path.exists(esm_path):
                raise FileNotFoundError(f"ESM embedding file not found: {esm_path}")
            # load embeddings: each line is one residue's 960-dim vector 1152 1536
            embeddings = np.loadtxt(esm_path)
            expected = len(self.index2resid[chain])
            if embeddings.shape[0] != expected or embeddings.shape[1] != 1152:
                raise ValueError(f"Embedding shape mismatch for {chain}: got {embeddings.shape}, expected ({expected},1152)")
            # map resid -> vector
            mapping = { self.index2resid[chain][i] : embeddings[i].tolist() for i in range(expected) }
            self.feature_esm[chain] = mapping

        # other preprocessing steps unchanged
        self.seq, self.index2resid = get_seq(self.pdb_id, self.protein_chains, dir_opts=self.dir_opts)
        self.nucleic_space = self._get_nucleic_acid_atom()
        self.surface_res = self._get_surface_res()
        self.interface = self._get_interface() if ligand_type in ['DNA','RNA','ATP','HEM'] else self._get_interface_mental(anno)
        self.site, self.site2point = self._get_site_biolip(anno)

    def _get_surface(self):
        extractPDB(self.pair, dir_opts=self.dir_opts)
        protonate(self.pair, dir_opts=self.dir_opts)
        extract_xyzrn(self.pair, dir_opts=self.dir_opts)
        probe_radius = 1.5 if self.ligand_type in ['DNA','RNA','ATP','HEM'] else 0.5
        vertices, faces, normalv, vert_info = computeMSMS(self.pair, dir_opts=self.dir_opts, probe_radius=probe_radius)
        mesh = pymesh.form_mesh(vertices, faces)
        vertices_new, faces_new, vert_info_new = fix_mesh(mesh, vert_info, 1.2)
        new_normalv = compute_normal(vertices_new, faces_new)
        mesh_new = pymesh.form_mesh(vertices_new, faces_new)
        mesh_new.add_attribute("vertex_nx"); mesh_new.set_attribute("vertex_nx", new_normalv[:,0])
        mesh_new.add_attribute("vertex_ny"); mesh_new.set_attribute("vertex_ny", new_normalv[:,1])
        mesh_new.add_attribute("vertex_nz"); mesh_new.set_attribute("vertex_nz", new_normalv[:,2])
        return mesh_new, new_normalv, vert_info_new

    def _load_pdb(self):
        pdb_file = os.path.join(self.dir_opts['raw_pdb_dir'], f"{self.pdb_id}.pdb")
        parser = MMCIFParser(QUIET=True)
        struct = parser.get_structure(self.pdb_id, pdb_file)
        return Selection.unfold_entities(struct, "M")[0]

    def _get_nucleic_acid_atom(self):
        if self.ligand_type == 'RNA':
            ligand_residue_names = ["RA", "RC", "RG", "RU"]
            ligand_symbol = 'NUC'
        elif self.ligand_type == 'DNA':
            ligand_residue_names = ["DA", "DC", "DG", "DT"]
            ligand_symbol = 'NUC'
        else:
            ligand_residue_names = [self.ligand_type,]
            ligand_symbol = self.ligand_type
        atom_list = []
        for chain_name in self.nucleic_chains.split('_'): #chain_name: e.g. A1. A is chain name, while 1 is kept for finding ligand pdb
            if len(chain_name)>1:
                pdb_file=os.path.join(self.dir_opts['ligand_dir'], self.pdb_id + '_' + ligand_symbol + '_' + chain_name[0] + '_' + chain_name[1] + '.pdb')
            else:
                pdb_file=os.path.join(self.dir_opts['ligand_dir'], self.pdb_id + '_' + ligand_symbol + '_' + chain_name[0] + '_1' + '.pdb')
            parser = PDBParser(QUIET=True)
            struct = parser.get_structure(self.pdb_id, pdb_file)
            model = Selection.unfold_entities(struct, "M")[0]
            chain = model.child_dict[chain_name[0]]
            for res in chain.child_list:
                res_type = res.resname.strip()
                if res_type not in ligand_residue_names:
                    continue
                for atom in res:
                    atom_list.append((atom.element, atom.coord, chain.id))
        return atom_list

    def _get_interface(self):
        NA_coords = np.array([atom[1] for atom in self.nucleic_space])
        kdtree = KDTree(NA_coords)
        interface = np.zeros([self.mesh.vertices.shape[0]])
        for index, vertex in enumerate(self.mesh.vertices):
            dis, indice = kdtree.query(vertex[None, :], k=1)
            if dis[0][0] < 3.0:
                interface[index] = 1
        return interface

    def _get_interface_mental(self, anno):
        anno = anno.split(':')
        site_by_biolip = set()
        for chain in anno:
            chain, binding_residues = chain.split(' ')[0], chain.split(' ')[1:]
            chain = chain[0]
            for residue in binding_residues:
                site_by_biolip.add(chain+'_'+ residue[1:])
        interface = []
        for item in self.vert_info:
            chain_res_id = item.split('_')[0] + '_' + item.split('_')[1]
            if chain_res_id in site_by_biolip:
                interface.append(1)
            else:
                interface.append(0)
        return interface

    def _get_site_biolip(self, anno):
        anno = anno.split(':')
        site_by_biolip = set()
        for chain in anno:
            chain, binding_residues = chain.split(' ')[0], chain.split(' ')[1:]
            chain = chain[0]
            for residue in binding_residues:
                site_by_biolip.add(chain+'_'+ residue[1:])
        self.site_by_biolip=site_by_biolip
        index2surface_res = {}
        surface_res2index = {}
        for index, res_id in enumerate(self.surface_res):
            index2surface_res[index] = res_id
            surface_res2index[res_id] = index
        residue2label = {surface_res2index[res_id]:[] for res_id in self.surface_res}
        point_res_id = np.zeros([len(self.vert_info)],dtype=np.long)
        for index, point_info in enumerate(self.vert_info):
            vert_resid = point_info.split('_')[0] +'_'+point_info.split('_')[1]
            point_res_id[index] = surface_res2index[vert_resid]
            if vert_resid in site_by_biolip:
                residue2label[surface_res2index[vert_resid]].append(1)
            else:
                residue2label[surface_res2index[vert_resid]].append(0)
        site_label = np.zeros([len(self.surface_res)],dtype=np.long)
        for index in range(len(self.surface_res)):
            site_label[index] = np.max(residue2label[index])
        return site_label, point_res_id

    def _get_surface_res(self):
        surface_res = set()
        for item in self.vert_info:
            chain = item.split('_')[0]
            resid = item.split('_')[1]
            surface_res.add(chain+'_'+resid)
        return list(surface_res)

    def _get_pKa_feature(self):
        # normalize pKa values to [0,1]
        mapping = {}
        for aa, (p1, p2, pr) in pKa_scale.items():
            n1 = (p1 - 1.82) / (2.38 - 1.82)
            n2 = (p2 - 8.80) / (10.96 - 8.80)
            n3 = (pr - 3.65) / (12.48 - 3.65)
            mapping[aa] = [n1, n2, n3]
        return mapping

    def _get_STED(self):
        #sum of total Euclidean distance
        STED = []
        for point in self.mesh.vertices:
            dis = np.sum(np.linalg.norm(self.mesh.vertices - point[None,:], axis=1))
            STED.append(dis)
        STED = np.array(STED)
        STED = (STED-np.min(STED))/(np.max(STED)-np.min(STED))
        self.STED_graph = STED[:,None]

    def _get_BOARD(self):
        tree = KDTree(self.mesh.vertices)
        indices = tree.query_radius(self.mesh.vertices, r=12)
        BOARD = []
        for index in range(self.mesh.vertices.shape[0]):
            points = indices[index]
            board_coord = self.mesh.vertices[points]
            signed_dis = np.mean((board_coord - self.mesh.vertices[index])*self.normalv[index])
            BOARD.append(signed_dis)
        BOARD = np.array(BOARD)
        BOARD = (BOARD-np.min(BOARD))/(np.max(BOARD)-np.min(BOARD))
        return BOARD[:, None]

    def _get_FLARE(self):
        tree = KDTree(self.mesh.vertices)
        indices_outer_ring = tree.query_radius(self.mesh.vertices,r=12)
        indices_inter_ring = tree.query_radius(self.mesh.vertices,r=9)
        FLARE = []
        for index in range(self.mesh.vertices.shape[0]):
            points_outer = set(indices_outer_ring[index])
            points_inter = set(indices_inter_ring[index])
            points = list(points_outer-points_inter)
            board_coord = self.mesh.vertices[points]
            signed_dis = np.mean((board_coord - self.mesh.vertices[index])*self.normalv[index])
            FLARE.append(signed_dis)
        FLARE = np.array(FLARE)
        FLARE = (FLARE-np.min(FLARE))/(np.max(FLARE)-np.min(FLARE))
        self.FLARE_graph = FLARE[:,None]


    def _get_graph(self):
        # build graphs: esm, atom_type, curvature
        atom_onehot = {"C":0, "H":1, "O":2, "N":3, "S":4}
        esm_graph = []
        atom_type_graph = []
        pka_graph = []

        for idx, vert in enumerate(self.vert_info):
            chain = vert.split('_')[0]
            res_id = int(vert.split('_')[1])
            # esm embedding 960-dim
            esm_graph.append(self.feature_esm[chain][res_id])
            # atom type
            atom_type = vert.split('_')[5]
            atom_type_graph.append([ atom_onehot.get(atom_type, 5) ])
            # pKa
            aa_three = vert.split('_')[3]
            pka_graph.append(self.pKa_mapping.get(aa_three, [0.5, 0.5, 0.5]))

        return np.array(esm_graph, dtype=float), \
               np.array(atom_type_graph, dtype=int), \
               np.array(pka_graph, dtype=float)


    def _tangent_vectors(self, scaler_function='curvature_graph', normals=None, tangent_bases=None):

        from pykeops.numpy import LazyTensor

        if normals is None:
            normals = self.normalv
        x, y, z= normals[..., 0], normals[..., 1], normals[..., 2]
        s=(2 * (z >= 0)) - 1.0  # = z.sign(), but =1. if z=0.
        a=-1 / (s + z)
        b=x * y * a
        uv=np.stack((1 + s * x * x * a, s * b, -s * x, b, s + y * y * a, -y), axis=-1)
        if tangent_bases is None:
            tangent_bases = uv.reshape(uv.shape[:1] + (2, 3))
        points = self.mesh.vertices / self.RQ_window
        # Normals and local areas:


        # 3. Steer the tangent bases according to the gradient of "weights" ----

        # 3.a) Encoding as KeOps LazyTensors:
        # Orientation scores:
        weights_j = LazyTensor(getattr(self, scaler_function)[:,0].reshape(1, -1, 1))  # (1, N, 1)

        # weights_j = LazyTensor(self.curvature_graph[:,0].reshape(1, -1, 1))  # (1, N, 1)
        # Vertices:
        x_i = LazyTensor(points[:, None, :])  # (N, 1, 3)
        x_j = LazyTensor(points[None, :, :])  # (1, N, 3)
        # Normals:
        n_i = LazyTensor(normals[:, None, :])  # (N, 1, 3)
        n_j = LazyTensor(normals[None, :, :])  # (1, N, 3)
        # Tangent basis:
        uv_i = LazyTensor(tangent_bases.reshape([-1, 1, 6]))  # (N, 1, 6)

        # 3.b) Pseudo-geodesic window:
        # Pseudo-geodesic squared distance:
        rho2_ij = ((x_j - x_i) ** 2).sum(-1) * ((2 - (n_i | n_j)) ** 2)  # (N, N, 1)
        # Gaussian window:
        alpha = 3.0
        window_ij = (1.0 + rho2_ij / alpha) ** (-alpha)  # (N, N, 1)

        # 3.c) Coordinates in the (u, v) basis - not oriented yet:
        X_ij = uv_i.matvecmult(x_j - x_i)  # (N, N, 2)

        # 3.d) Local average in the tangent plane:
        orientation_weight_ij = window_ij * weights_j  # (N, N, 1)
        orientation_vector_ij = orientation_weight_ij * X_ij  # (N, N, 2)


        orientation_vector_i = orientation_vector_ij.sum(dim=1)  # (N, 2)
        orientation_vector_i = (
            orientation_vector_i + 1e-5
        )  # Just in case someone's alone...

        # 3.e) Normalize stuff:
        # orientation_vector_i = F.normalize(orientation_vector_i, p=2, dim=-1)  #  (N, 2)
        orientation_vector_i = orientation_vector_i / np.linalg.norm(orientation_vector_i,axis=1,keepdims=True)
        ex_i, ey_i = (
            orientation_vector_i[:, 0][:, None],
            orientation_vector_i[:, 1][:, None],
        )  # (N,1)

        # 3.f) Re-orient the (u,v) basis:
        uv_i = tangent_bases  # (N, 2, 3)
        u_i, v_i = uv_i[:, 0, :], uv_i[:, 1, :]  # (N, 3)
        tangent_bases = np.stack(
            (ex_i * u_i + ey_i * v_i, -ey_i * u_i + ex_i * v_i), axis=1
        )  # (N, 6)

        # 4. Store the local 3D frame as an attribute --------------------------
        nuv = np.concatenate(
            (normals.reshape(-1, 1, 3), tangent_bases), axis=1
        )
        return nuv

    def get_data(self):
        self.esm_graph, self.atom_type_graph, self.pka_graph = self._get_graph()
        self.BOARD_graph = self._get_BOARD()
        self.nuv_BOARD = self._tangent_vectors(scaler_function='BOARD_graph')
        data_label = {
            'xyz': self.mesh.vertices,
            'nuv': self.nuv_BOARD,
            'y': self.interface,
            'site': self.site,
            'site_point': self.site2point,
            'esm': self.esm_graph,
            'pka': self.pka_graph,
            'xyz_type': self.atom_type_graph
        }
        # print(data_label)
        save_path = os.path.join(self.dir_opts['data_label'], self.pair)
        with open(save_path, 'wb') as pid:
            cPickle.dump(data_label, pid)

if __name__ == '__main__':
    # 6lnh:C:C1	C_C H170 D172 H223
    from default_config.dir_options import dir_opts
    dir_opts = dir_opts('Dataset/FE/')
    proc = MetalBindProcessor('6lnh:C:C1', anno='C_C H170 D172 H223', ligand_type='FE', dir_opts=dir_opts)
    proc.get_data()