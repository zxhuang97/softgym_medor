import glob
import pdb

import h5py
import numpy as np
import random
import pyflex
from softgym.envs.cloth_env import ClothEnv
from copy import deepcopy
from softgym.utils.misc import vectorized_range, vectorized_meshgrid, load_cloth, find_2D_rigid_trans, \
    get_closest_point
from softgym.utils.pyflex_utils import center_object
from scipy.spatial.transform import Rotation as R
import pickle
import os


class AnyClothFlattenEnv(ClothEnv):
    def __init__(self, cloth_type=None, obj_paths=None, cloth3d_dir="", collect_mode=None,
                 load_flat=False, ambiguity_agnostic=False,
                 cached_states_path="", **kwargs):
        """
        Args:
            cloth_type: specify cloth type for cached states collection or testing.
            cloth3d_dir: specify the directory of cloth3d dataset
            obj_paths: specify a self-defined set of objs, primarily for testing purpose
            collect_mode: flat, random_drop
            load_flat: load flat cloth states
            ambiguity_agnostic: whether to use ambiguity agnostic cannonical states. Only used for
                                task canonicalization
            **kwargs:
        """
        print('Initializing AnyClothFlattening environment')
        self.collect_mode = collect_mode  # flatten or random_drop
        self.cloth_type = cloth_type
        self.cached_states_path = cached_states_path
        self.ambiguity_agnostic = ambiguity_agnostic
        super(AnyClothFlattenEnv, self).__init__(**kwargs)

        assert (cloth_type is not None and cloth3d_dir is not None) or obj_paths is not None

        self.cloth3d_dir = cloth3d_dir
        if obj_paths is not None:
            self.obj_paths = obj_paths
        elif cloth_type not in ['square', 'rectangle']:
            self.obj_paths = glob.glob(
                f"{self.cloth3d_dir}/mesh/{cloth_type}/*.obj") if obj_paths is None else obj_paths
            self.obj_paths = sorted(self.obj_paths)
        else:
            self.obj_paths = [i for i in range(500)]

        self.flat_states_path = f"{self.cloth3d_dir}/mesh/{cloth_type}/flat_states.pkl"
        # models in cloth3D are in human T-pose. We need to flatten them by dropping.
        if load_flat:

            print('Try to load flat states from ', self.flat_states_path)
            if os.path.exists(self.flat_states_path):
                with open(self.flat_states_path, "rb") as handle:
                    self.flatten_configs, self.flatten_states = pickle.load(handle)
            else:
                self.collect_flatten_state()
        else:
            self.flatten_configs, self.flatten_states = None, None
        self.get_cached_configs_and_states(cached_states_path, self.num_variations)
        self.prev_covered_area = None  # Should not be used until initialized

    def generate_env_variation(self, num_variations=1, **kwargs):
        """ Generate initial states. """
        if self.flatten_configs is None:
            print("No flattened cloth states found. You need to generate flatten states of .")

        max_lift_step = 100  # Maximum number of steps waiting for the cloth to stablize
        max_drop_step = 300
        stable_vel_threshold = 0.1  # Cloth stable when all particles' vel are smaller than this
        generated_configs, generated_states = [], []
        default_config = self.get_default_config()

        # The variations are splitted into 9:1 sets
        train_num = num_variations * 9 // 10
        # Cloth models are also splitted into 9:1 sets
        clothes_num = len(self.flatten_configs)
        train_cloth_num = clothes_num * 9 // 10
        if self.eval_flag:  # generating cached states for testing
            train_num = 0
        for i in range(num_variations):
            # TODO: random flip to remove the bias in the dataset
            if i < train_num:
                random_id = np.random.randint(train_cloth_num)
            else:
                random_id = np.random.randint(train_cloth_num, clothes_num)
            while 1:
                config = deepcopy(self.flatten_configs[random_id])
                flat_state = deepcopy(self.flatten_states[random_id])
                config.update(default_config)
                config['cloth_id'] = random_id
                flat_state['camera_params'] = config['camera_params']
                self.update_camera(config['camera_name'], config['camera_params'][config['camera_name']])
                self.set_scene(config, state=flat_state)
                flatten_cov = config['flatten_area']

                if self.collect_mode == "random_drop":
                    self.action_tool.reset([0., -1., 0.])
                    pos = pyflex.get_positions().reshape(-1, 4)

                    num_particle = pos.shape[0]
                    pickpoint = random.randint(0, num_particle - 1)
                    curr_pos = pyflex.get_positions()
                    original_inv_mass = curr_pos[pickpoint * 4 + 3]
                    # Set the mass of the pickup point to infinity so that it generates enough force to the rest of the cloth
                    curr_pos[pickpoint * 4 + 3] = 0
                    pyflex.set_positions(curr_pos)
                    pickpoint_pos = curr_pos[
                                    pickpoint * 4: pickpoint * 4 + 3].copy()  # Pos of the pickup point is fixed to this point
                    # pickpoint_pos[1] += np.random.random(1) * 0.5 + 0.5
                    tgt_height = np.random.random(1) * 1 + 0.5

                    # Pick up the cloth and wait to stablize
                    init_height = pickpoint_pos[1]
                    speed = 0.02
                    steps = int((tgt_height - init_height) / speed)
                    for j in range(steps):
                        curr_pos = pyflex.get_positions()
                        curr_vel = pyflex.get_velocities()
                        pickpoint_pos[1] = (tgt_height - init_height) * (j * speed) + init_height
                        curr_pos[pickpoint * 4: pickpoint * 4 + 3] = pickpoint_pos
                        curr_pos[pickpoint * 4 + 3] = 0
                        curr_vel[pickpoint * 3: pickpoint * 3 + 3] = [0, 0, 0]
                        pyflex.set_positions(curr_pos)
                        pyflex.set_velocities(curr_vel)
                        pyflex.step(render=False)
                    print('lift', steps)
                    for j in range(max_lift_step):
                        curr_pos = pyflex.get_positions()
                        curr_vel = pyflex.get_velocities()
                        curr_pos[pickpoint * 4: pickpoint * 4 + 3] = pickpoint_pos
                        curr_vel[pickpoint * 3: pickpoint * 3 + 3] = [0, 0, 0]
                        pyflex.set_positions(curr_pos)
                        pyflex.set_velocities(curr_vel)
                        pyflex.step(render=False)
                        if np.alltrue(np.abs(curr_vel) < stable_vel_threshold) and j > 5:
                            break

                    # Drop the cloth and wait to stablize
                    curr_pos = pyflex.get_positions()
                    curr_pos[pickpoint * 4 + 3] = original_inv_mass
                    pyflex.set_positions(curr_pos)
                    for j in range(max_drop_step):
                        # obs = self._get_obs()
                        pyflex.step()
                        curr_vel = np.abs(pyflex.get_velocities())
                        if np.alltrue(curr_vel < stable_vel_threshold) and j > 10:
                            break
                init_pos = pyflex.get_positions().reshape(-1, 4)[:, :3]
                cur_cov = self._get_current_covered_area(pyflex.get_positions())
                if not self.eval_flag or cur_cov / config['flatten_area'] < 0.7:
                    config['flatten_area'] = flatten_cov
                    config['init_area'] = cur_cov
                    break
            center_object()

            generated_configs.append(deepcopy(config))
            generated_states.append(deepcopy(self.get_state()))

            # Generate goal canonical state for the task of canonicalization
            if self.ambiguity_agnostic:
                canon_poses = self.collect_ambiguity_aware_canon_states(random_id)
            else:
                canon_poses = self.flatten_states[random_id]['particle_pos'].reshape(-1, 4)[None, ...,
                              :3]
            generated_states[-1]['canon_poses'] = canon_poses
            init_canon_dis, _ = self.compute_canon_dis_ambiguity_agnostic(canon_poses,
                                                                          init_pos,
                                                                          find_rigid=False)

            init_canon_dis_rigid, _ = self.compute_canon_dis_ambiguity_agnostic(canon_poses,
                                                                                init_pos,
                                                                                find_rigid=True)
            generated_states[-1]['init_canon_dis'] = init_canon_dis
            generated_states[-1]['init_canon_dis_rigid'] = init_canon_dis_rigid
            print('config {}: cloth id {} particle radius {}, '
                  'flatten area: {}, normalized coverage {}'.format(i,
                                                                    config['cloth_id'],
                                                                    config['radius'],
                                                                    generated_configs[-1][
                                                                        'flatten_area'],
                                                                    cur_cov / config['flatten_area']))

        return generated_configs, generated_states

    def get_all_obs(self):
        rgbd = self.get_rgbd(show_picker=False)
        rgbd = np.array(rgbd).reshape(self.camera_height, self.camera_width, 4)
        depth = rgbd[:, :, 3]
        img = self.render(mode='rgb_array')
        rgb = img.astype(np.uint8)
        coords = pyflex.get_positions().reshape(-1, 4)

        return coords, rgb, depth

    def collect_ambiguity_aware_canon_states(self, cloth_id):
        """
        Return a list of valid flatten states to account for ambiguity
        Trousers: 180 rotation
        Tshirt: None
        Dress: 180 rotation
        Skirt: 360 rotation > 12 bins
        Jumpsuit: 180 rotation
        Args:
            cloth_id:  id of the cloth being processed

        Returns:
            canon_poses: a numpy array of KxNx3, where K is the number of possible ambiguity

        """

        print('Start to collect canonical poses! Cloth id ', cloth_id)
        obj = self.obj_paths[cloth_id]
        config = self.get_default_config(super_gravity=True)

        # rotate it so it looks right in top-down camera view
        v, f, stretch_e, bend_e, shear_e = load_cloth(obj)
        v = self.rotate_particles([180, 180, 0],
                                  v)  # first 180 rotate around up, second 180 rotate around z
        v = self.translate_particles([0, 0.0, 0], v)

        # iterate over the category-dependent ambiguity
        if self.cloth_type in ['Trousers', 'Dress', 'Jumpsuit']:
            vs = np.stack([self.rotate_particles([180 * i, 0, 0], v) for i in range(2)])
        elif self.cloth_type == 'Skirt':
            num_bins = 12
            vs = np.stack(
                [self.rotate_particles([360 / num_bins * i, 0, 0], v) for i in range(num_bins)])
        else:
            vs = v[None]
        vs = [self.translate_particles([0, 0.0, 0], v) for v in vs]
        canon_poses = []

        # using gravity to flatten the cloth
        for v in vs:
            config['v'], config['f'], config['stretch_e'], config['bend_e'], config[
                'shear_e'] = v, f, stretch_e, bend_e, shear_e
            self.update_camera(config['camera_name'], config['camera_params'][config['camera_name']])
            self.set_scene(config)
            for _ in range(150):
                pyflex.step(render=True)
            canon_poses.append(pyflex.get_positions().reshape(-1, 4)[..., :3])
        return np.stack(canon_poses)

    def _reset(self, canon_pos=None, init_pos=None, init_covered_area=-1):
        """ Right now only use one initial state"""
        self.prev_covered_area = self._get_current_covered_area(pyflex.get_positions())
        if hasattr(self, 'action_tool'):
            curr_pos = pyflex.get_positions()
            cx, cy = self._get_center_point(curr_pos)
            self.action_tool.reset([cx, -1, cy])
        pyflex.step()

        cur_state = None if self.current_config_id is None else self.cached_init_states[
            self.current_config_id]

        if init_pos is not None:
            self.init_pos = init_pos
        else:
            self.init_pos = pyflex.get_positions().reshape(-1, 4)[:, :3]

        if canon_pos is not None:
            self.canon_poses = canon_pos
        elif cur_state is not None and 'canon_poses' in cur_state:
            self.canon_poses = cur_state['canon_poses'][:, :, :3]  # KxNx3
            if not self.ambiguity_agnostic:  # if not ambiguity agnostic, use the first pose
                self.canon_poses = self.canon_poses[0:1]
        else:
            print('Warnings! Flat states are not loaded and canon dis computation is wrong')
            self.canon_poses = pyflex.get_positions().reshape(1, -1, 4)[:, :, :3]
        self.init_covered_area = init_covered_area
        if self.init_covered_area < 0:
            self.init_covered_area = self.prev_covered_area
        if "init_canon_dis" not in self.current_config:
            self.init_canon_dis, _ = self.compute_canon_dis_ambiguity_agnostic(self.canon_poses,
                                                                               self.init_pos,
                                                                               find_rigid=False)

            self.init_canon_dis_rigid, _ = self.compute_canon_dis_ambiguity_agnostic(self.canon_poses,
                                                                                     self.init_pos,
                                                                                     find_rigid=True)
        else:
            self.init_canon_dis = self.current_config["init_canon_dis"]
            self.init_canon_dis_rigid = self.current_config["init_canon_dis_rigid"]
        return self._get_obs()

    def _step(self, action):
        # action could be a tuple of (picker_action, per_particle_residual)
        self.action_tool.step(action)
        if self.action_mode in ['sawyer', 'franka']:
            pyflex.step(self.action_tool.next_action)
        elif not isinstance(action, tuple) or action[
            1] is None:  # if align pc, step has already been called
            pyflex.step()
        return

    def _get_current_covered_area(self, pos):
        """
        Calculate the covered area by taking max x,y cood and min x,y coord, create a discritized grid between the points
        :param pos: Current positions of the particle states
        """
        pos = np.reshape(pos, [-1, 4])
        min_x = np.min(pos[:, 0])
        min_y = np.min(pos[:, 2])
        max_x = np.max(pos[:, 0])
        max_y = np.max(pos[:, 2])
        init = np.array([min_x, min_y])
        span = np.array([max_x - min_x, max_y - min_y]) / 100.
        pos2d = pos[:, [0, 2]]

        offset = pos2d - init
        slotted_x_low = np.maximum(
            np.round((offset[:, 0] - self.cloth_particle_radius) / span[0]).astype(int), 0)
        slotted_x_high = np.minimum(
            np.round((offset[:, 0] + self.cloth_particle_radius) / span[0]).astype(int), 100)
        slotted_y_low = np.maximum(
            np.round((offset[:, 1] - self.cloth_particle_radius) / span[1]).astype(int), 0)
        slotted_y_high = np.minimum(
            np.round((offset[:, 1] + self.cloth_particle_radius) / span[1]).astype(int), 100)
        # Method 1
        grid = np.zeros(10000)  # Discretization
        listx = vectorized_range(slotted_x_low, slotted_x_high)
        listy = vectorized_range(slotted_y_low, slotted_y_high)
        listxx, listyy = vectorized_meshgrid(listx, listy)
        idx = listxx * 100 + listyy
        idx = np.clip(idx.flatten(), 0, 9999)
        grid[idx] = 1

        return np.sum(grid) * span[0] * span[1]

    def _get_center_point(self, pos):
        pos = np.reshape(pos, [-1, 4])
        min_x = np.min(pos[:, 0])
        min_y = np.min(pos[:, 2])
        max_x = np.max(pos[:, 0])
        max_y = np.max(pos[:, 2])
        return 0.5 * (min_x + max_x), 0.5 * (min_y + max_y)

    def compute_reward(self, action=None, obs=None, set_prev_reward=False):
        particle_pos = pyflex.get_positions()
        curr_covered_area = self._get_current_covered_area(particle_pos)
        r = curr_covered_area
        return r

    @staticmethod
    def compute_canon_dis(pts1, pts2, find_rigid=False):
        """
        pts1: target  pts2: predicted
        If find_rigid is true, then find the optimal rigid transformation that align pts1 with pts2
        Return distance to canonical state, and goal position(potentially after transformation)
        """
        new_pts1 = pts1.copy()
        is_registered = pts1.shape[0] == pts2.shape[0]
        if is_registered:
            if find_rigid:
                rigid_mat2d = find_2D_rigid_trans(pts1, pts2)
                new_pts1[:, [0, 2]] = pts1[:, [0, 2]] @ rigid_mat2d[:, :2].T + rigid_mat2d[:, 2].T
            return np.linalg.norm(new_pts1 - pts2, axis=1).mean(), new_pts1
        else:
            # run icp for 5 iterations can compute chamfer
            for i in range(5):
                nn_idx = get_closest_point(pts2, new_pts1)
                nn_pts1 = new_pts1[nn_idx]
                rigid_mat2d = find_2D_rigid_trans(nn_pts1, pts2)
                new_pts1[:, [0, 2]] = new_pts1[:, [0, 2]] @ rigid_mat2d[:, :2].T + rigid_mat2d[:, 2].T
            nn_idx = get_closest_point(pts2, new_pts1)
            nn_pts1 = new_pts1[nn_idx]
            return np.linalg.norm(nn_pts1 - pts2, axis=1).mean(), new_pts1

    @staticmethod
    def compute_canon_dis_ambiguity_agnostic(pts_tgts, pts, find_rigid=False):
        """Given a list of targets, compute the best fit canon distance"""
        cur_canon_dis_list, canon_tgt_list = [], []
        for pts_tgt in pts_tgts:
            cur_canon_dis, canon_tgt = AnyClothFlattenEnv.compute_canon_dis(pts_tgt, pts,
                                                                            find_rigid=find_rigid)
            cur_canon_dis_list.append(cur_canon_dis)
            canon_tgt_list.append(canon_tgt)

        best_id = np.argmin(cur_canon_dis_list)
        cur_canon_dis_best, canon_tgt_best = cur_canon_dis_list[best_id], canon_tgt_list[best_id]
        return cur_canon_dis_best, canon_tgt_best

    def flip(self, pts):
        new_pts = pts.copy()
        center = pts.mean(0)
        new_pts -= center
        new_pts = new_pts * np.array([[-1, -1, 1]])
        return new_pts

    def _get_info(self):
        """
        Compute metrics for evaluation:
        1. coverage
        2. L2 norm distance to canonical pose
        3. L2 norm distance to canonical pose with optimal translation
        4. l2 norm distance to canonical pose with optimal rigid transformation
        Returns:
        """
        particle_pos = pyflex.get_positions().reshape(-1, 4)
        curr_covered_area = self._get_current_covered_area(particle_pos)
        init_covered_area = curr_covered_area if self.init_covered_area is None else self.init_covered_area
        max_covered_area = self.get_current_config()['flatten_area']
        canon_poses = self.canon_poses
        num_particles = canon_poses.shape[1]
        # if self.ambiguity_agnostic:
        canon_metric = self.compute_canon_dis_ambiguity_agnostic  # if self.ambiguity_agnostic else self.compute_canon_dis
        # else:
        #     canon_metric = self.compute_canon_dis
        #     self.canon_poses = canon_poses  # dirty
        # select the first k particles according  the number of particle in cloth mesh
        # because sometimes(for visualization), we may have dummy particles that makes shape of particle_pos > canon_pos
        particle_pos = particle_pos[:num_particles, :3]
        # init_pos = self.init_pos[:num_particles, :3]

        # init_canon_dis, _ = canon_metric(canon_poses, init_pos, find_rigid=False)
        cur_canon_dis, canon_tgt = canon_metric(canon_poses, particle_pos, find_rigid=False)

        # init_canon_dis_rigid, _ = canon_metric(canon_poses, init_pos, find_rigid=True)
        cur_canon_dis_rigid, canon_rigid_tgt = canon_metric(canon_poses, particle_pos, find_rigid=True)

        info = {
            'coverage': curr_covered_area,
            'normalized_coverage': curr_covered_area / max_covered_area,
            'normalized_coverage_improvement': (curr_covered_area - init_covered_area) / (
                    max_covered_area - init_covered_area),

            'canon_dis': cur_canon_dis,
            'canon_dis_rigid': cur_canon_dis_rigid,
            'normalized_canon_improvement': 1 - cur_canon_dis / self.init_canon_dis,
            'normalized_canon_improvement_rigid': 1 - cur_canon_dis_rigid / self.init_canon_dis_rigid,

            'canon_tgt': canon_tgt,
            'canon_rigid_tgt': canon_rigid_tgt,
        }
        if 'qpg' in self.action_mode:
            info['total_steps'] = self.action_tool.total_steps
        return info

    def get_picked_particle(self):
        pps = np.ones(shape=self.action_tool.num_picker,
                      dtype=np.int32) * -1  # -1 means no particles picked
        for i, pp in enumerate(self.action_tool.picked_particles):
            if pp is not None:
                pps[i] = pp
        return pps

    def get_picked_particle_new_position(self):
        intermediate_picked_particle_new_pos = self.action_tool.intermediate_picked_particle_pos
        if len(intermediate_picked_particle_new_pos) > 0:
            return np.vstack(intermediate_picked_particle_new_pos)
        else:
            return []

    def set_scene(self, config, state=None):
        if self.render_mode == 'particle':
            render_mode = 1
        elif self.render_mode == 'cloth':
            render_mode = 2
        elif self.render_mode == 'both':
            render_mode = 3
        camera_params = config['camera_params'][config['camera_name']]
        env_idx = 6
        if 'fov' not in config:
            config['fov'] = 45
        scene_params = np.concatenate(
            [[config['damping'], config['dyn_fric'], config['particle_fric'], config['gravity'],
              config['fov'],
              config['vel']], config['stiff'][:],
             [config['mass'], config['radius']],
             camera_params['pos'][:], camera_params['angle'][:],
             [camera_params['width'], camera_params['height']],
             [render_mode], config['cloth_size'][:],
             [config['v'].shape[0], config['f'].shape[0],
              config['stretch_e'].shape[0],
              config['shear_e'].shape[0],
              config['bend_e'].shape[0]],
             config['v'].flatten(), config['f'].flatten(), config['stretch_e'].flatten(),
             config['shear_e'].flatten(),
             config['bend_e'].flatten()])
        robot_params = []
        pyflex.set_scene(env_idx, scene_params, 0)
        self.default_pos = pyflex.get_positions()

        if state is not None:
            self.set_state(state)

        self.current_config = deepcopy(config)

    def get_default_config(self, super_gravity=False):
        # cam_pos, cam_angle = np.array([0.0, 0.82, 0.00]), np.array([0, -np.pi/2., 0.])
        cam_pos, cam_angle = np.array([-0.0, 0.82, 0.82]), np.array([0, -45 / 180. * np.pi, 0.])
        g_scale = 20 if super_gravity else 1
        config = {
            'damping': np.random.uniform(1, 1),
            'dyn_fric': np.random.uniform(1.2, 1.2),
            'particle_fric': np.random.uniform(1.2, 1.2),
            'gravity': -9.8 * g_scale,
            'rot': 0.0,
            'vel': 0.0,
            'stiff': [np.random.uniform(1.2, 1.2), np.random.uniform(0.6, 0.6), np.random.uniform(1, 1)],
            # Stretch, Bend and Shear
            'mass': np.random.uniform(0.0003, 0.0003),
            'radius': self.cloth_particle_radius,  # / 1.8,
            'camera_name': self.camera_name,
            'camera_params': {'default_camera':
                                  {'pos': cam_pos,
                                   'angle': cam_angle,
                                   'width': self.camera_width,
                                   'height': self.camera_height},
                              'top_down_camera': {
                                  'pos': np.array([-0.0, 0.65, 0]),
                                  'angle': np.array([0, -90 / 180 * np.pi, 0]),
                                  'width': self.camera_width,
                                  'height': self.camera_height
                              },
                              'bottom_side_camera': {
                                  'pos': np.array([0.0, 0.1, 0.4]),
                                  'angle': np.array([0, -0 / 180 * np.pi, 0]),
                                  'width': self.camera_width,
                                  'height': self.camera_height
                              }
                              },
            'cloth_type': self.cloth_type,
            'cloth_size': [-1, -1]
        }

        return config

    def rotate_particles(self, angle, pos):
        r = R.from_euler('zyx', angle, degrees=True)
        new_pos = pos.copy()[:, :3]
        center = np.mean(new_pos, axis=0)
        new_pos -= center
        new_pos = r.apply(new_pos)
        new_pos += center
        return new_pos

    def translate_particles(self, new_pos, pos):
        """ Translate the cloth so that it lies on the ground with center at pos """
        center = np.mean(pos, axis=0)
        center[1] = np.min(pos, axis=0)[1]
        pos[:, :3] -= center[:3]
        pos[:, :3] += np.asarray(new_pos)
        return pos

    # def _get_flat_pos(self):
    #     cloth_id = self.current_config['cloth_id']
    #     assert self.flatten_states is not None
    #     flat_state = deepcopy(self.flatten_states[cloth_id])
    #     flat_pos = flat_state['particle_pos'].reshape((-1, 4))[:, :3]
    #     flat_pos = self.rotate_particles([0, 180, 0],
    #                                      flat_pos)  # first 180 rotate around up, second 180 rotate around z
    #
    #     return flat_pos

    def collect_flatten_state(self):
        print(f"Start to collect flatten cloth model for {self.cloth_type}")

        cloth_dir = f"{self.cloth3d_dir}/{self.cloth_type}"

        generated_configs, generated_states = [], []

        for i, obj in enumerate(self.obj_paths):
            print(f"collecting {i} flatten states")
            config = self.get_default_config(super_gravity=True)
            if self.cloth_type not in ["square", "rectangle"]:
                v, f, stretch_e, bend_e, shear_e = load_cloth(obj)
                v = self.rotate_particles([180, 0, 0], v)
                v = self.translate_particles([0, 0.0, 0], v)
                config["cloth_size"] = [-1, -1]
            else:
                v, f, stretch_e, bend_e, shear_e = np.array([]), np.array([]), np.array([]), np.array(
                    []), np.array([])
                if self.cloth_type == "square":
                    dim = np.random.randint(30, 50)
                    config["cloth_size"] = [dim, dim]
                else:
                    config["cloth_size"] = [np.random.randint(25, 35), np.random.randint(45, 55)]

            config['v'], config['f'], config['stretch_e'], config['bend_e'], config[
                'shear_e'] = v, f, stretch_e, bend_e, shear_e

            self.update_camera(config['camera_name'], config['camera_params'][config['camera_name']])
            self.set_scene(config)

            if self.cloth_type not in ["square", "rectangle"]:
                for _ in range(150):
                    pyflex.step()
            else:
                self.set_to_flatten(config)

            flatten_cov = self._get_current_covered_area(pyflex.get_positions())
            generated_configs.append(deepcopy(config))
            generated_states.append(deepcopy(self.get_state()))
            # self.current_config = config  # Needed in _set_to_flatten function
            generated_configs[-1]['flatten_area'] = flatten_cov
            generated_configs[-1]['gravity'] = -9.8
            generated_configs[-1]['cloth_id'] = i

        with open(self.flat_states_path, "wb") as handle:
            pickle.dump((generated_configs, generated_states), handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Flatten states  of {len(generated_states)} cloths are saved to {self.flat_states_path}")
        self.flatten_configs, self.flatten_states = generated_configs, generated_states


if __name__ == '__main__':
    from softgym.registered_env import env_arg_dict
    from softgym.registered_env import SOFTGYM_ENVS
    import copy
    import cv2


    def prepare_policy(env):
        print("preparing policy! ", flush=True)

        # move one of the picker to be under ground
        shape_states = pyflex.get_shape_states().reshape(-1, 14)
        shape_states[1, :3] = -1
        shape_states[1, 7:10] = -1

        # move another picker to be above the cloth
        pos = pyflex.get_positions().reshape((-1, 4))[:, :3]
        pp = np.random.randint(len(pos))
        shape_states[0, :3] = pos[pp] + [0., 0.06, 0.]
        shape_states[0, 7:10] = pos[pp] + [0., 0.06, 0.]
        pyflex.set_shape_states(shape_states.flatten())


    env_args = copy.deepcopy(env_arg_dict[env_name])
    env_args['render_mode'] = 'cloth'
    env_args['observation_mode'] = 'cam_rgb'
    env_args['render'] = True
    env_args['camera_height'] = 720
    env_args['camera_width'] = 720
    env_args['camera_name'] = 'default_camera'
    env_args['headless'] = True
    env_args['action_repeat'] = 1
    env_args['picker_radius'] = 0.01
    env_args['picker_threshold'] = 0.00625
    # env_args['cached_states_path'] = 'tshirt_flatten_init_states_1.pkl'
    env_args['num_variations'] = 1
    env_args['use_cached_states'] = False
    env_args['save_cached_states'] = False
    env_args['cloth_type'] = 'tshirt-small'
    # pkl_path = './softgym/cached_initial_states/shorts_flatten.pkl'

    env = SOFTGYM_ENVS[env_name](**env_args)
    env.reset(config_id=0)
    # env._set_to_flat()
    # env.move_to_pos([0, 0.1, 0])
    # pyflex.step()
    # i = 0
    # import pickle

    # while (1):
    #     pyflex.step(render=True)
    #     if i % 500 == 0:
    #         print('saving pkl to ' + pkl_path)
    #         pos = pyflex.get_positions()
    #         with open(pkl_path, 'wb') as f:
    #             pickle.dump(pos, f)
    #     i += 1
    #     print(i)

    obs = env._get_obs()
    cv2.imwrite('./small_tshirt.png', obs)
    # cv2.imshow('obs', obs)
    # cv2.waitKey()

    prepare_policy(env)

    particle_positions = pyflex.get_positions().reshape(-1, 4)[:, :3]
    n_particles = particle_positions.shape[0]
    # p_idx = np.random.randint(0, n_particles)
    # p_idx = 100
    pos = particle_positions
    ok = False
    while not ok:
        pp = np.random.randint(len(pos))
        if np.any(np.logical_and(
                np.logical_and(np.abs(pos[:, 0] - pos[pp][0]) < 0.00625,
                               np.abs(pos[:, 2] - pos[pp][2]) < 0.00625),
                pos[:, 1] > pos[pp][1])):
            ok = False
        else:
            ok = True
    picker_pos = particle_positions[pp] + [0, 0.01, 0]

    timestep = 50
    movement = np.random.uniform(0, 1, size=(3)) * 0.4 / timestep
    movement = np.array([0.2, 0.2, 0.2]) / timestep
    action = np.zeros((timestep, 8))
    action[:, 3] = 1
    action[:, :3] = movement

    shape_states = pyflex.get_shape_states().reshape((-1, 14))
    shape_states[1, :3] = -1
    shape_states[1, 7:10] = -1

    shape_states[0, :3] = picker_pos
    shape_states[0, 7:10] = picker_pos

    pyflex.set_shape_states(shape_states)
    pyflex.step()

    obs_list = []

    for a in action:
        obs, _, _, _ = env.step(a)
        obs_list.append(obs)
        # cv2.imshow("move obs", obs)
        # cv2.waitKey()

    for t in range(30):
        a = np.zeros(8)
        obs, _, _, _ = env.step(a)
        obs_list.append(obs)
        # cv2.imshow("move obs", obs)
        # cv2.waitKey()

    from softgym.utils.visualization import save_numpy_as_gif

    save_numpy_as_gif(np.array(obs_list), '{}.gif'.format(
        env_args['cloth_type']
    ))
